"""Financial QA Agent — LangGraph-based agent loop with parse-driven routing."""

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .config import settings
from .tools.knowledge_base import fetch_knowledge
from .tools.market_data import fetch_market_data
from .tools.news_search import fetch_news


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

    The parse node writes parse_result. Each fetch tool writes to its own
    field. The synthesize node reads all non-empty fields.
    """

    question: str
    parse_result: ParseResult
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
- time_period: Map to yfinance period strings. \
today/now → 1d, this week → 5d, last month/1 month → 1mo, \
last 3 months/last quarter → 3mo, last 6 months → 6mo, \
last year/1 year → 1y, 2 years → 2y, 5 years → 5y, \
year to date → ytd, all time → max. \
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
    llm = _build_llm()
    prompt = PARSE_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = _strip_markdown_fencing(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: empty parse result, all tools will use raw question
        return {"parse_result": {}}

    return {"parse_result": parsed}


async def fetch_market_data_node(state: AgentState) -> dict:
    """Fetch structured market data (OHLCV, fundamentals)."""
    result = await fetch_market_data(state["question"], state.get("parse_result", {}))
    return {"market_data": result}


async def fetch_news_node(state: AgentState) -> dict:
    """Fetch recent financial news via Brave Search."""
    result = await fetch_news(state["question"], state.get("parse_result", {}))
    return {"news_data": result}


async def fetch_knowledge_node(state: AgentState) -> dict:
    """Fetch knowledge from local vector DB (with web fallback)."""
    result = await fetch_knowledge(state["question"], state.get("parse_result", {}))
    return {"knowledge_data": result}


SYNTHESIZE_PROMPT = """\
You are a financial analyst assistant. Based on the data provided below, \
give a clear, accurate, and helpful answer to the user's question.

If the data is limited or unavailable, acknowledge that and provide \
what insight you can. Do not fabricate data.

## User Question
{question}

## Available Data
{context}

Provide your answer:"""


async def synthesize_node(state: AgentState) -> dict:
    """Synthesize a final answer from all available tool outputs."""
    llm = _build_llm()

    context_parts: list[str] = []
    if state.get("market_data"):
        context_parts.append(f"### Market Data\n{state['market_data']}")
    if state.get("news_data"):
        context_parts.append(f"### Recent News\n{state['news_data']}")
    if state.get("knowledge_data"):
        context_parts.append(f"### Knowledge Base\n{state['knowledge_data']}")

    context = "\n\n".join(context_parts) if context_parts else "No data was retrieved."

    prompt = SYNTHESIZE_PROMPT.format(question=state["question"], context=context)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"answer": response.content.strip()}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_by_parse_result(state: AgentState) -> list[str] | str:
    """Route to fetch node(s) based on what the parse step extracted.

    - tickers present → fetch_market_data
    - needs_news → fetch_news
    - knowledge_queries present → fetch_knowledge
    - nothing extracted → fetch_knowledge (fallback for general questions)
    """
    pr = state.get("parse_result", {})
    nodes: list[str] = []

    if pr.get("tickers"):
        nodes.append("fetch_market_data")
    if pr.get("needs_news"):
        nodes.append("fetch_news")
    if pr.get("knowledge_queries"):
        nodes.append("fetch_knowledge")

    if not nodes:
        return "fetch_knowledge"

    return nodes if len(nodes) > 1 else nodes[0]


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and compile the agent graph.

    Topology:
        parse → (conditional) → fetch_market_data ──┐
                               → fetch_news ──────────┤→ synthesize → END
                               → fetch_knowledge ─────┘
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
            "market_data": "",
            "news_data": "",
            "knowledge_data": "",
            "answer": "",
        }

        result = await self._graph.ainvoke(initial_state)
        return result["answer"]
