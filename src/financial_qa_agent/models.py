"""Pydantic models for structured data flowing through the financial QA agent.

This module formalizes all plain-dict data structures used by tools, the agent
pipeline, and the SSE trace protocol. LangGraph state types (AgentState,
ParseResult) remain as TypedDict in agent.py — these models are companions
for validation and serialization, not replacements.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Market Data (tools/market_data.py)
# ---------------------------------------------------------------------------


class HistoryRecord(BaseModel):
    """Single OHLCV data point from yfinance history."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TickerData(BaseModel):
    """Fundamental and price data for a single ticker."""

    model_config = ConfigDict(populate_by_name=True)

    ticker: str
    name: str
    current_price: float | None = None
    market_cap: int | None = None
    pe_ratio: float | None = None
    w52_high: float | None = Field(default=None, alias="52w_high")
    w52_low: float | None = Field(default=None, alias="52w_low")
    sector: str | None = None
    industry: str | None = None
    recent_history: list[HistoryRecord] = []


# ---------------------------------------------------------------------------
# Knowledge Base (tools/knowledge_base.py)
# ---------------------------------------------------------------------------


class KnowledgeResult(BaseModel):
    """A single knowledge base or web search result."""

    text: str
    source: str = "local"
    title: str = ""


# ---------------------------------------------------------------------------
# News Search (tools/news_search.py)
# ---------------------------------------------------------------------------


class NewsResult(BaseModel):
    """A single news search result from Brave API."""

    title: str = "No title"
    description: str = "No description"
    url: str = ""
    age: str = ""


# ---------------------------------------------------------------------------
# Parse Result (companion to TypedDict in agent.py)
# ---------------------------------------------------------------------------


VALID_TIME_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})


class ParseResultModel(BaseModel):
    """Pydantic companion for the ParseResult TypedDict.

    Validates LLM JSON output and provides defaults. Convert to dict
    with .model_dump() for LangGraph state compatibility.
    """

    tickers: list[str] = []
    company_names: list[str] = []
    time_period: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    asset_type: str | None = None
    sector: str | None = None
    needs_news: bool = False
    news_query: str | None = None
    knowledge_queries: list[str] = []

    @field_validator("time_period")
    @classmethod
    def validate_time_period(cls, v: str | None) -> str | None:
        """Coerce invalid time_period values to None.

        The LLM is instructed to return only valid yfinance periods, but may
        occasionally hallucinate values like '2w' or '4mo'. Invalid values are
        silently dropped (None) so the downstream tool applies its default.
        """
        if v is not None and v not in VALID_TIME_PERIODS:
            return None
        return v


# ---------------------------------------------------------------------------
# Trace Events (agent.py SSE protocol)
# ---------------------------------------------------------------------------


class TraceEvent(BaseModel):
    """Agent pipeline stage progress event."""

    event_type: Literal["trace"] = "trace"
    stage: str
    status: str
    detail: str = ""
    data: dict | None = None
    tools: list[str] | None = None


class ToolInputEvent(BaseModel):
    """Tool invocation input event."""

    event_type: Literal["tool_input"] = "tool_input"
    tool: str
    input: dict


class ToolOutputEvent(BaseModel):
    """Tool invocation output event."""

    event_type: Literal["tool_output"] = "tool_output"
    tool: str
    output: str


class AnswerEvent(BaseModel):
    """Final answer event."""

    event_type: Literal["answer"] = "answer"
    answer: str


class ErrorEvent(BaseModel):
    """Error event."""

    event_type: Literal["error"] = "error"
    message: str
