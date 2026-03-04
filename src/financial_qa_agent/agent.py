"""Financial QA Agent — LangGraph-based agent loop with parse-driven routing."""

import asyncio
import contextvars
import json
import logging
from datetime import date
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
from .tools.knowledge_base import fetch_local_knowledge, fetch_web_knowledge
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
    which tools fire and which synthesize prompt to use.
    """

    question_type: str  # "analysis" or "knowledge" — LLM determines user intent
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

    The parse node writes parse_result. Routing determines the path.
    Each tool node writes to its own field. The synthesize node reads
    all available data to produce the final answer.

    Knowledge path: fetch_local_knowledge → evaluate_sufficiency →
    [fetch_web_knowledge if needed] → synthesize.
    """

    question: str
    parse_result: ParseResult
    question_type: str  # "analysis" or "knowledge" (from parse LLM, for tracing)
    market_data: str
    news_data: str
    local_knowledge_data: str  # ChromaDB local results
    web_knowledge_data: str  # Brave web search results
    knowledge_sufficient: str  # "yes" | "no" (LLM evaluation)
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
structured entities.

Extraction rules:
- question_type: Determine the user's intent — "analysis" or "knowledge". \
"analysis" means the user wants data-driven market analysis: price checks, \
performance comparisons, trend analysis, technical analysis, portfolio review. \
The answer will present computed facts from OHLCV data + analytical interpretation. \
"knowledge" means the user wants explanations, reasons, concepts, mechanisms, \
or contextual understanding — even if specific assets are mentioned. \
Examples: "AAPL过去一年涨了多少" → analysis (wants price change data). \
"黄金最近两月涨的原因是什么" → knowledge (wants explanation of why gold rose). \
"What is Apple's P/E ratio?" → analysis (wants a specific data point). \
"Why did Tesla stock drop?" → knowledge (wants causal explanation). \
"Compare AAPL and MSFT" → analysis (wants data comparison). \
"How do stock options work?" → knowledge (wants concept explanation). \
"What caused the 2008 financial crisis?" → knowledge (wants historical explanation).
- tickers: Resolve to yfinance-compatible symbols. \
Company names to ticker (Apple → AAPL, Tesla → TSLA, Bank of America → BAC). \
Crypto to -USD format (Bitcoin → BTC-USD, Ethereum → ETH-USD). \
Forex to =X format (EUR/USD → EURUSD=X). \
Indices to ^ format (S&P 500 → ^GSPC, Dow Jones → ^DJI, Nasdaq → ^IXIC). \
Commodities to yfinance futures format \
(gold/黄金 → GC=F, silver/白银 → SI=F, crude oil/原油 → CL=F, \
Brent oil/布伦特原油 → BZ=F, natural gas/天然气 → NG=F, \
copper/铜 → HG=F, platinum/铂金 → PL=F, palladium/钯金 → PA=F, \
corn/玉米 → ZC=F, wheat/小麦 → ZW=F, soybean/大豆 → ZS=F, \
coffee/咖啡 → KC=F, sugar/糖 → SB=F, cotton/棉花 → CT=F). \
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
- time_start / time_end: Use INSTEAD of time_period when the user asks about \
a specific date or explicit date range. Format MUST be YYYY-MM-DD. \
For a single date like "January 15" or "1/15", set time_start to that date \
and time_end to the NEXT day (e.g. time_start="2026-01-15", time_end="2026-01-16"). \
For date ranges like "from March 1 to March 10", set both accordingly. \
When the user says a date without a year, assume the most recent occurrence \
(past, not future) relative to today. If time_start is set, time_period MUST be null. \
Examples: "data on 1/15" → time_start="2026-01-15", time_end="2026-01-16", time_period=null. \
"AAPL from Feb 1 to Feb 14" → time_start="2026-02-01", time_end="2026-02-14", time_period=null. \
"last Monday" → compute the date, set time_start=that date, time_end=next day, time_period=null.
- asset_type: equity, etf, crypto, forex, index, sector, or commodity. Null if unclear.
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

Today's date: {today}

Question: {question}

Respond with ONLY valid JSON, no markdown fencing:
{{"question_type": "analysis"|"knowledge", "tickers": [...], "company_names": [...], \
"time_period": ..., "time_start": ..., "time_end": ..., "asset_type": ..., \
"sector": ..., "needs_news": ..., "news_query": ..., "knowledge_queries": [...]}}"""


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
    prompt = PARSE_PROMPT.format(question=state["question"], today=date.today().isoformat())
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


# ---------------------------------------------------------------------------
# Knowledge Hub Nodes (local → evaluate → optional web)
# ---------------------------------------------------------------------------


async def fetch_local_knowledge_node(state: AgentState) -> dict:
    """Fetch knowledge from local ChromaDB vector database."""
    pr = state.get("parse_result", {})
    input_data = {
        "knowledge_queries": pr.get("knowledge_queries", []),
        "question": state["question"],
    }
    await _emit_trace(
        **TraceEvent(
            stage="fetch_local_knowledge", status="started",
            detail="Querying local knowledge base...",
        ).model_dump()
    )
    await _emit_trace(**ToolInputEvent(tool="local_knowledge", input=input_data).model_dump())

    result = await fetch_local_knowledge(state["question"], pr)

    await _emit_trace(**ToolOutputEvent(tool="local_knowledge", output=result).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="fetch_local_knowledge", status="completed",
            detail="Local knowledge query complete",
        ).model_dump()
    )
    return {"local_knowledge_data": result}


EVALUATE_SUFFICIENCY_PROMPT = """\
You are a knowledge sufficiency evaluator. Decide whether the local knowledge \
base results below are sufficient to comprehensively answer the user's question.

Evaluation criteria:
- Are the results directly relevant to the question?
- Is there enough detail to provide a complete, accurate answer?
- Would a web search likely add meaningful information?

If results are empty, sparse, or off-topic, mark as NOT sufficient.
If results cover the question well with enough detail, mark as sufficient.

## User Question
{question}

## Local Knowledge Results
{local_results}

Respond with ONLY valid JSON, no markdown fencing:
{{"sufficient": true, "reason": "brief explanation"}}"""


async def evaluate_sufficiency_node(state: AgentState) -> dict:
    """LLM evaluates whether local knowledge results suffice.

    Returns ``knowledge_sufficient`` as ``"yes"`` or ``"no"``.
    On parse failure, conservatively defaults to ``"no"`` (trigger web search).
    """
    local_data = state.get("local_knowledge_data", "")
    await _emit_trace(
        **TraceEvent(
            stage="evaluate_sufficiency", status="started",
            detail="Evaluating local knowledge sufficiency...",
        ).model_dump()
    )

    llm = _build_llm()
    prompt = EVALUATE_SUFFICIENCY_PROMPT.format(
        question=state["question"],
        local_results=local_data or "No results found.",
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = _strip_markdown_fencing(response.content)

    try:
        evaluation = json.loads(raw)
        is_sufficient = evaluation.get("sufficient", False)
        reason = evaluation.get("reason", "")
    except json.JSONDecodeError:
        logger.debug("evaluate_sufficiency: JSON decode failed, defaulting to no")
        is_sufficient = False
        reason = "Evaluation parse failed — defaulting to web search"

    decision = "yes" if is_sufficient else "no"
    detail = f"Sufficient: {decision} — {reason}"

    logger.debug("evaluate_sufficiency: %s", detail)
    await _emit_trace(
        **TraceEvent(
            stage="evaluate_sufficiency", status="completed",
            detail=detail,
            data={"sufficient": decision, "reason": reason},
        ).model_dump()
    )
    return {"knowledge_sufficient": decision}


async def fetch_web_knowledge_node(state: AgentState) -> dict:
    """Fetch knowledge from web search and store in ChromaDB."""
    pr = state.get("parse_result", {})
    input_data = {
        "knowledge_queries": pr.get("knowledge_queries", []),
        "news_query": pr.get("news_query"),
        "question": state["question"],
    }
    await _emit_trace(
        **TraceEvent(
            stage="fetch_web_knowledge", status="started",
            detail="Searching web for additional knowledge...",
        ).model_dump()
    )
    await _emit_trace(**ToolInputEvent(tool="web_knowledge", input=input_data).model_dump())

    result = await fetch_web_knowledge(state["question"], pr)

    await _emit_trace(**ToolOutputEvent(tool="web_knowledge", output=result).model_dump())
    await _emit_trace(
        **TraceEvent(
            stage="fetch_web_knowledge", status="completed",
            detail="Web knowledge search complete",
        ).model_dump()
    )
    return {"web_knowledge_data": result}


# ---------------------------------------------------------------------------
# Synthesize Prompts
# ---------------------------------------------------------------------------


SYNTHESIZE_ANALYSIS_PROMPT = """\
You are a financial analyst assistant. Based on the market data and news \
provided below, answer the user's question about the specified assets or markets.

Your answer MUST use the following section format with markdown headers. \
Each section starts with a ## header on its own line, followed by the content.

## Fact
The available data includes raw OHLCV (Open/High/Low/Close/Volume) rows and \
some snapshot fields (current price, market cap, P/E, 52-week range, etc.). \
The snapshot fields are point-in-time values, NOT historical — do NOT treat \
them as historical data points (e.g. 52-week low is NOT "the price one year ago"). \
To answer questions about price changes, trends, or performance over a period, \
you MUST compute the relevant metrics yourself from the OHLCV rows \
(e.g. compare the first and last close prices in the history range, \
calculate percentage change, identify highs/lows within the range).

Structure this section in two parts:

**Key Metrics** — Present the computed figures that directly answer the user's question \
as a concise bullet list. Only include question-relevant metrics. Be precise with numbers. \
Examples: start/end prices, percentage change, period high/low, average volume, P/E ratio.

**Summary** — One sentence that directly answers the user's question in plain language. \
If the question involves price movement, explicitly state the trend direction: \
upward (上涨/bullish), downward (下跌/bearish), or sideways/oscillating (震荡/range-bound). \
Include the key number (e.g. "rose 12.3%" or "fell $15 to $142"). \
This summary should let the reader grasp the answer at a glance before reading the Analysis.

Do NOT dump all OHLCV rows — extract and compute what's needed.

## Analysis
Provide your analytical interpretation of the facts above. Directly address \
the user's question — don't just describe the data, explain what it means. \
Discuss trends, comparisons, implications, and context. Connect the facts \
to a clear takeaway that answers the user's question.

## References
Look at the "Recent News" section in the available data below. \
If news articles are present, list each news source as a bullet point: \
- [Source Title](URL)
If no "Recent News" section exists in the available data, omit this \
section entirely (do not include the ## References header).

Do not fabricate data or URLs. If data is limited, acknowledge that in the analysis.

---
User Question: {question}

Available Data:
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

---
User Question: {question}

Available Data:
{context}

Provide your sectioned answer:"""


async def synthesize_node(state: AgentState) -> dict:
    """Synthesize a final answer from all available tool outputs.

    Reads question type from ``parse_result.question_type`` which is set by the
    parse LLM based on user intent. This is more accurate than inferring from
    data presence, since knowledge questions can also have market data as
    supporting context (e.g. "Why did gold rise?" has tickers but wants
    an explanation, not a data analysis).
    """
    pr = state.get("parse_result", {})
    question_type = pr.get("question_type", "knowledge")
    sources = []
    if state.get("market_data"):
        sources.append("market_data")
    if state.get("news_data"):
        sources.append("news")
    if state.get("local_knowledge_data"):
        sources.append("local_knowledge")
    if state.get("web_knowledge_data"):
        sources.append("web_knowledge")
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
    if state.get("local_knowledge_data"):
        context_parts.append(f"### Local Knowledge\n{state['local_knowledge_data']}")
    if state.get("web_knowledge_data"):
        context_parts.append(f"### Web Search Results\n{state['web_knowledge_data']}")

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
    """Route to fetch node(s) based on LLM-determined question type.

    The parse node determines ``question_type`` ("analysis" or "knowledge").
    Routing combines question type with available tickers and news flags:

    - **Analysis**: fetch_market_data (always) + fetch_news (if needs_news)
    - **Knowledge with tickers**: fetch_market_data + fetch_local_knowledge
      [+ fetch_news] — market data provides supporting context for the answer
    - **Knowledge without tickers**: fetch_local_knowledge only

    Note: ``question_type`` is set on the state dict for trace detail, but
    LangGraph does not propagate routing-function mutations to downstream
    nodes. The synthesize node reads ``question_type`` from ``parse_result``.
    """
    pr = state.get("parse_result", {})
    qtype = pr.get("question_type", "knowledge")

    if qtype == "analysis" and pr.get("tickers"):
        # Type A: Data-driven analysis
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

    elif qtype == "knowledge" and pr.get("tickers"):
        # Type B: Knowledge question with asset context (market data as support)
        state["question_type"] = "knowledge"
        nodes = ["fetch_market_data", "fetch_local_knowledge"]
        if pr.get("needs_news"):
            nodes.append("fetch_news")

        _emit_trace_sync(
            **TraceEvent(
                stage="route", status="completed",
                detail=f"Question type: knowledge (with tickers) | Routing to: {', '.join(nodes)}",
                tools=nodes,
            ).model_dump()
        )
        return nodes

    else:
        # Type C: Pure knowledge Q/A (local → evaluate → optional web)
        state["question_type"] = "knowledge"
        _emit_trace_sync(
            **TraceEvent(
                stage="route", status="completed",
                detail="Question type: knowledge | Routing to: fetch_local_knowledge",
                tools=["fetch_local_knowledge"],
            ).model_dump()
        )
        return "fetch_local_knowledge"


def route_after_sufficiency(state: AgentState) -> str:
    """Route based on knowledge sufficiency evaluation.

    - ``"yes"`` → synthesize (local results are enough)
    - ``"no"``  → fetch_web_knowledge (need web search)
    """
    decision = state.get("knowledge_sufficient", "no")

    if decision == "yes":
        _emit_trace_sync(
            **TraceEvent(
                stage="route_sufficiency", status="completed",
                detail="Local knowledge sufficient — skipping web search",
            ).model_dump()
        )
        return "synthesize"
    else:
        _emit_trace_sync(
            **TraceEvent(
                stage="route_sufficiency", status="completed",
                detail="Local knowledge insufficient — fetching from web",
            ).model_dump()
        )
        return "fetch_web_knowledge"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the agent graph.

    Topology (LLM-determined question type + knowledge hub):
        parse → question_type?
          → Analysis (tickers):  fetch_market_data [+ fetch_news] → synthesize → END
          → Knowledge (tickers): fetch_market_data [+ fetch_news] + fetch_local_knowledge
                                   → evaluate_sufficiency → [fetch_web_knowledge?] → synthesize → END
          → Knowledge (no tickers): fetch_local_knowledge → evaluate_sufficiency
                                      → [fetch_web_knowledge?] → synthesize → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("fetch_market_data", fetch_market_data_node)
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("fetch_local_knowledge", fetch_local_knowledge_node)
    graph.add_node("evaluate_sufficiency", evaluate_sufficiency_node)
    graph.add_node("fetch_web_knowledge", fetch_web_knowledge_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.set_entry_point("parse")

    # Conditional routing from parse → fetch node(s)
    graph.add_conditional_edges(
        "parse",
        route_by_parse_result,
        ["fetch_market_data", "fetch_news", "fetch_local_knowledge"],
    )

    # Analysis path: fetch nodes → synthesize
    graph.add_edge("fetch_market_data", "synthesize")
    graph.add_edge("fetch_news", "synthesize")

    # Knowledge hub path: local → evaluate → conditional → synthesize
    graph.add_edge("fetch_local_knowledge", "evaluate_sufficiency")
    graph.add_conditional_edges(
        "evaluate_sufficiency",
        route_after_sufficiency,
        ["synthesize", "fetch_web_knowledge"],
    )
    graph.add_edge("fetch_web_knowledge", "synthesize")

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
            "local_knowledge_data": "",
            "web_knowledge_data": "",
            "knowledge_sufficient": "",
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
                "local_knowledge_data": "",
                "web_knowledge_data": "",
                "knowledge_sufficient": "",
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
