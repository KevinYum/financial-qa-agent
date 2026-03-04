"""Financial QA Agent — LangGraph-based agent loop with parse-driven routing."""

import asyncio
import contextvars
import json
import logging
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .config import settings
from .models import (
    AnswerEvent,
    ErrorEvent,
    ParseResultModel,
    ToolInputEvent,
    ToolOutputEvent,
    TraceEvent,
)
from .tools.knowledge_base import fetch_knowledge
from .tools.market_data import fetch_market_data
from .tools.news_search import fetch_news

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace infrastructure (contextvars — no-op when unused)
# ---------------------------------------------------------------------------

_trace_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "_trace_queue", default=None
)


async def _emit_trace(event_type: str, **kwargs) -> None:
    """Emit a trace event to the active queue, if any.

    When called from ``ask_with_trace``, events flow to the SSE endpoint.
    When called from ``ask`` (no queue set), this is a silent no-op.
    """
    queue = _trace_queue.get(None)
    if queue is not None:
        await queue.put({"event_type": event_type, **kwargs})


def _emit_trace_sync(event_type: str, **kwargs) -> None:
    """Non-async variant for synchronous functions (e.g. routing)."""
    queue = _trace_queue.get(None)
    if queue is not None:
        queue.put_nowait({"event_type": event_type, **kwargs})


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class ParseResult(TypedDict, total=False):
    """Structured output from the parse node.

    The parse LLM extracts these from the user's question. Fields drive
    which tools fire — no classification label needed.
    """

    tickers: list[str]  # yfinance symbols: ["AAPL", "BTC-USD", "^GSPC"]
    company_names: list[str]  # Original names: ["Apple", "Bitcoin"]
    time_period: str | None  # yfinance period: "1d", "5d", "1mo", "3mo", etc.
    time_start: str | None  # Explicit date range start (YYYY-MM-DD)
    time_end: str | None  # Explicit date range end (YYYY-MM-DD)
    asset_type: str | None  # "equity", "etf", "crypto", "forex", "index", "sector"
    sector: str | None  # Sector name when asset_type is "sector"
    needs_news: bool  # Whether to fetch news for this question
    news_query: str | None  # Refined search query for Brave
    knowledge_queries: list[str]  # Conceptual sub-questions for knowledge base


class AgentState(TypedDict):
    """State that flows through the agent graph.

    The parse node writes parse_result. Routing sets question_type.
    Each fetch tool writes to its own field. The synthesize node reads
    question_type to select the answer format.
    """

    question: str
    parse_result: ParseResult
    question_type: str  # "analysis" (tickers present) or "knowledge" (no tickers)
    market_data: str
    news_data: str
    knowledge_data: str
    answer: str


def _build_llm() -> ChatOpenAI:
    """Build the LLM client from settings.

    Supports OpenAI and OpenRouter — detected by base_url.
    """
    kwargs: dict = {
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "model": settings.llm_model,
        "temperature": 0.2,
    }
    # OpenRouter requires extra headers
    if "openrouter" in settings.llm_base_url:
        kwargs["default_headers"] = {
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_x_title,
        }
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

PARSE_PROMPT = """\
You are a financial question parser. Analyze the question and extract \
structured entities. Do NOT classify — just extract what's in the question.

Extraction rules:
- tickers: Resolve to yfinance-compatible symbols. \
Company names to ticker (Apple → AAPL, Tesla → TSLA, Bank of America → BAC). \
Crypto to -USD format (Bitcoin → BTC-USD, Ethereum → ETH-USD). \
Forex to =X format (EUR/USD → EURUSD=X). \
Indices to ^ format (S&P 500 → ^GSPC, Dow Jones → ^DJI, Nasdaq → ^IXIC). \
Sector queries with no specific stock to sector ETF \
(technology → XLK, healthcare → XLV, financials → XLF, energy → XLE, \
consumer discretionary → XLY, consumer staples → XLP, industrials → XLI, \
materials → XLB, utilities → XLU, real estate → XLRE, communication services → XLC). \
Empty list if no financial instruments mentioned.
- company_names: Original human-readable names mentioned.
- time_period: MUST be one of these exact yfinance values: \
1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. \
No other values are valid — do NOT invent periods like 2w, 2d, 4mo, etc. \
Always pick the smallest valid period that fully covers the requested range. \
Examples: today/now → 1d, this week/past few days → 5d, \
past 2 weeks/half month/last 2-3 weeks → 1mo, \
last month/1 month/past 4 weeks → 1mo, \
last 2 months/past 6-8 weeks → 3mo, \
last quarter/3 months → 3mo, last 6 months/past half year → 6mo, \
last 9 months → 1y, last year/1 year/past 12 months → 1y, \
2 years/18 months → 2y, 3-5 years → 5y, \
6-10 years/decade → 10y, year to date → ytd, all time/max → max. \
If multiple tickers have different time spans, use the longest. \
Null if no time reference.
- time_start / time_end: For explicit date ranges (YYYY-MM-DD). \
Use instead of time_period when user specifies exact dates.
- asset_type: equity, etf, crypto, forex, index, or sector. Null if unclear.
- sector: Sector name if asset_type is sector.
- needs_news: true if the question asks about recent events, earnings, \
market sentiment, breaking news, or anything time-sensitive. \
If multiple tickers and at least one needs news, set true. \
false for purely data or conceptual questions.
- news_query: A concise search query for web search if needs_news is true. \
Remove analytical framing, keep entity names and event keywords. Null if not needed.
- knowledge_queries: List of conceptual or educational sub-questions \
(e.g., "What is compound interest?", "How do stock options work?"). \
Empty list if the question is purely about data or news.

Question: {question}

Respond with ONLY valid JSON, no markdown fencing:
{{"tickers": [...], "company_names": [...], "time_period": ..., \
"time_start": ..., "time_end": ..., "asset_type": ..., "sector": ..., \
"needs_news": ..., "news_query": ..., "knowledge_queries": [...]}}"""


def _strip_markdown_fencing(raw: str) -> str:
    """Strip markdown code fences from LLM output if present."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        else:
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return stripped


async def parse_node(state: AgentState) -> dict:
    """Parse the question into structured entities via LLM."""
    await _emit_trace(
        **TraceEvent(
            stage="parse", status="started",
            detail="Extracting entities from question...",
        ).model_dump()
    )
    logger.debug("parse_node: question=%r", state["question"])

    llm = _build_llm()
    prompt = PARSE_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = _strip_markdown_fencing(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("parse_node: JSON decode failed, falling back to empty result")
        await _emit_trace(
            **TraceEvent(
                stage="parse", status="completed",
                detail="Parse failed — falling back to empty result",
                data={},
            ).model_dump()
        )
        return {"parse_result": {}}

    # Validate and normalize via Pydantic model (provides defaults for missing fields)
    validated = ParseResultModel(**parsed)
    result_dict = validated.model_dump()

    detail = (
        f"Extracted {len(validated.tickers)} ticker(s)"
        + (f": {', '.join(validated.tickers)}" if validated.tickers else "")
        + (f" | needs_news={validated.needs_news}" if validated.needs_news else "")
        + (f" | {len(validated.knowledge_queries)} knowledge query(s)" if validated.knowledge_queries else "")
    )
    logger.debug("parse_node: result=%s", result_dict)
    await _emit_trace(
        **TraceEvent(
            stage="parse", status="completed",
            detail=detail, data=result_dict,
        ).model_dump()
    )
    return {"parse_result": result_dict}


async def fetch_market_data_node(state: AgentState) -> dict:
    """Fetch structured market data (OHLCV, fundamentals)."""
    pr = state.get("parse_result", {})
    input_data = {
        "tickers": pr.get("tickers", []),
        "time_period": pr.get("time_period"),
        "time_start": pr.get("time_start"),
        "time_end": pr.get("time_end"),
        "asset_type": pr.get("asset_type"),
        "sector": pr.get("sector"),
    }
    await _emit_trace(
        **TraceEvent(
            stage="fetch_market_data", status="started",
            detail="Fetching market data...",
        ).model_dump()
    )
    await _emit_trace(**ToolInputEvent(tool="market_data", input=input_data).model_dump())

    result = await fetch_market_data(state["question"], pr)

    await _emit_trace(**ToolOutputEvent(tool="market_data", output=result).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="fetch_market_data", status="completed",
            detail="Market data fetched",
        ).model_dump()
    )
    return {"market_data": result}


async def fetch_news_node(state: AgentState) -> dict:
    """Fetch recent financial news via Brave Search."""
    pr = state.get("parse_result", {})
    input_data = {
        "news_query": pr.get("news_query"),
        "question": state["question"],
    }
    await _emit_trace(
        **TraceEvent(
            stage="fetch_news", status="started",
            detail="Searching for news...",
        ).model_dump()
    )
    await _emit_trace(**ToolInputEvent(tool="news_search", input=input_data).model_dump())

    result = await fetch_news(state["question"], pr)

    await _emit_trace(**ToolOutputEvent(tool="news_search", output=result).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="fetch_news", status="completed",
            detail="News search complete",
        ).model_dump()
    )
    return {"news_data": result}


async def fetch_knowledge_node(state: AgentState) -> dict:
    """Fetch knowledge from local vector DB (with web fallback)."""
    pr = state.get("parse_result", {})
    input_data = {
        "knowledge_queries": pr.get("knowledge_queries", []),
        "question": state["question"],
    }
    await _emit_trace(
        **TraceEvent(
            stage="fetch_knowledge", status="started",
            detail="Querying knowledge base...",
        ).model_dump()
    )
    await _emit_trace(**ToolInputEvent(tool="knowledge_base", input=input_data).model_dump())

    result = await fetch_knowledge(state["question"], pr)

    await _emit_trace(**ToolOutputEvent(tool="knowledge_base", output=result).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="fetch_knowledge", status="completed",
            detail="Knowledge base query complete",
        ).model_dump()
    )
    return {"knowledge_data": result}


SYNTHESIZE_ANALYSIS_PROMPT = """\
You are a financial analyst assistant. Based on the market data and news \
provided below, answer the user's question about the specified assets or markets.

Your answer MUST use the following section format with markdown headers. \
Each section starts with a ## header on its own line, followed by the content.

## Fact
Present the key factual data in a structured, readable format. Include current \
prices, price changes, fundamentals (market cap, P/E ratio, 52-week range), \
and any relevant metrics from the market data. Be precise with numbers. \
Use line breaks for readability when presenting multiple data points.

## Analysis
Provide your analytical interpretation of the data. Discuss trends, comparisons, \
implications, and context. What do the numbers suggest? What patterns are notable?

## References
If news data is available, list each news source as a bullet point: \
- [Source Title](URL)
If no news data is available, omit this section entirely (do not include the header).

Do not fabricate data or URLs. If data is limited, acknowledge that in the analysis.

## User Question
{question}

## Available Data
{context}

Provide your sectioned answer:"""


SYNTHESIZE_KNOWLEDGE_PROMPT = """\
You are a financial education assistant. Based on the knowledge base results \
provided below, answer the user's question clearly and accurately.

Your answer MUST use the following section format with markdown headers. \
Each section starts with a ## header on its own line, followed by the content.

## Answer
Provide a clear, comprehensive answer to the question. Explain concepts, \
mechanisms, and relationships. Use plain language that is easy to understand.

## References
List the sources used to compile this answer. For each source that has a URL, \
include it as a bullet point: \
- [Source Title](URL)
If a source has no URL (local knowledge base), omit it from this list. \
If no sources have URLs, write "Based on general financial knowledge."

Do not fabricate information or URLs.

## User Question
{question}

## Available Data
{context}

Provide your sectioned answer:"""


async def synthesize_node(state: AgentState) -> dict:
    """Synthesize a final answer from all available tool outputs.

    Determines question type from the data actually present in the state:
    - market_data non-empty → "analysis" (structured data + analytical interpretation)
    - otherwise → "knowledge" (answer + references with source URLs)

    This is more reliable than reading ``question_type`` from state, because
    LangGraph routing functions don't propagate direct state mutations to
    downstream nodes.
    """
    # Derive question type from which tools actually produced data
    question_type = "analysis" if state.get("market_data") else "knowledge"
    sources = []
    if state.get("market_data"):
        sources.append("market_data")
    if state.get("news_data"):
        sources.append("news")
    if state.get("knowledge_data"):
        sources.append("knowledge_base")
    await _emit_trace(
        **TraceEvent(
            stage="synthesize", status="started",
            detail=f"Generating {question_type} answer from: {', '.join(sources) or 'no data'}",
        ).model_dump()
    )

    llm = _build_llm()

    context_parts: list[str] = []
    if state.get("market_data"):
        context_parts.append(f"### Market Data\n{state['market_data']}")
    if state.get("news_data"):
        context_parts.append(f"### Recent News\n{state['news_data']}")
    if state.get("knowledge_data"):
        context_parts.append(f"### Knowledge Base\n{state['knowledge_data']}")

    context = "\n\n".join(context_parts) if context_parts else "No data was retrieved."

    # Select prompt based on question type
    if question_type == "analysis":
        prompt_template = SYNTHESIZE_ANALYSIS_PROMPT
    else:
        prompt_template = SYNTHESIZE_KNOWLEDGE_PROMPT

    prompt = prompt_template.format(question=state["question"], context=context)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    await _emit_trace(**AnswerEvent(answer=answer).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="synthesize", status="completed",
            detail="Answer generated",
        ).model_dump()
    )
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_by_parse_result(state: AgentState) -> list[str] | str:
    """Route to fetch node(s) based on question type.

    Two mutually exclusive paths determined by whether tickers are present:

    - **Analysis** (tickers found): fetch_market_data (always) + fetch_news (if needs_news)
    - **Knowledge** (no tickers): fetch_knowledge

    Note: ``question_type`` is set on the state dict for trace detail, but
    LangGraph does not propagate routing-function mutations to downstream
    nodes. The synthesize node independently derives question type from which
    tool data is present.
    """
    pr = state.get("parse_result", {})

    if pr.get("tickers"):
        # Type A: Analysis for assets/markets
        state["question_type"] = "analysis"
        nodes = ["fetch_market_data"]
        if pr.get("needs_news"):
            nodes.append("fetch_news")

        _emit_trace_sync(
            **TraceEvent(
                stage="route", status="completed",
                detail=f"Question type: analysis | Routing to: {', '.join(nodes)}",
                tools=nodes,
            ).model_dump()
        )
        return nodes if len(nodes) > 1 else nodes[0]
    else:
        # Type B: General knowledge Q/A
        state["question_type"] = "knowledge"
        _emit_trace_sync(
            **TraceEvent(
                stage="route", status="completed",
                detail="Question type: knowledge | Routing to: fetch_knowledge",
                tools=["fetch_knowledge"],
            ).model_dump()
        )
        return "fetch_knowledge"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the agent graph.

    Topology (two-type routing):
        parse → tickers? → Analysis:  fetch_market_data [+ fetch_news] → synthesize → END
                         → No tickers: fetch_knowledge                 → synthesize → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("fetch_market_data", fetch_market_data_node)
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("fetch_knowledge", fetch_knowledge_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.set_entry_point("parse")

    # Conditional routing from parse → fetch node(s)
    graph.add_conditional_edges(
        "parse",
        route_by_parse_result,
        ["fetch_market_data", "fetch_news", "fetch_knowledge"],
    )

    # All fetch nodes converge to synthesize
    graph.add_edge("fetch_market_data", "synthesize")
    graph.add_edge("fetch_news", "synthesize")
    graph.add_edge("fetch_knowledge", "synthesize")

    # Synthesize leads to end
    graph.add_edge("synthesize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API (interface preserved from stub)
# ---------------------------------------------------------------------------


class FinancialQAAgent:
    """Agent that answers financial questions using a LangGraph pipeline.

    Public interface: ``await agent.ask(question) -> str``
    This is the same signature as the original stub, keeping main.py unchanged.
    """

    def __init__(self) -> None:
        self._graph = build_graph()

    async def ask(self, question: str) -> str:
        """Process a financial question and return an answer."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        initial_state: AgentState = {
            "question": question.strip(),
            "parse_result": {},
            "question_type": "",
            "market_data": "",
            "news_data": "",
            "knowledge_data": "",
            "answer": "",
        }

        result = await self._graph.ainvoke(initial_state)
        return result["answer"]

    async def ask_with_trace(
        self, question: str, trace_queue: asyncio.Queue
    ) -> str:
        """Process a financial question with trace events streamed to *trace_queue*.

        Each event is a dict with ``event_type`` plus payload fields.
        A ``None`` sentinel is always put on the queue when done (success or error).
        """
        if not question or not question.strip():
            await trace_queue.put(
                ErrorEvent(message="Question cannot be empty").model_dump()
            )
            await trace_queue.put(None)
            raise ValueError("Question cannot be empty")

        _trace_queue.set(trace_queue)
        try:
            initial_state: AgentState = {
                "question": question.strip(),
                "parse_result": {},
                "question_type": "",
                "market_data": "",
                "news_data": "",
                "knowledge_data": "",
                "answer": "",
            }
            result = await self._graph.ainvoke(initial_state)
            return result["answer"]
        except Exception as exc:
            await trace_queue.put(
                ErrorEvent(message=str(exc)).model_dump()
            )
            raise
        finally:
            await trace_queue.put(None)  # sentinel: signals stream end
            _trace_queue.set(None)
