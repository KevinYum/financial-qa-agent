"""Unit tests for Pydantic models."""

import pytest

from src.financial_qa_agent.models import (
    AnswerEvent,
    ErrorEvent,
    FundamentalData,
    HistoryRecord,
    IncomeStatementRecord,
    KnowledgeResult,
    NewsResult,
    ParseResultModel,
    TickerData,
    ToolInputEvent,
    ToolOutputEvent,
    TraceEvent,
)


# ---------------------------------------------------------------------------
# Market Data Models
# ---------------------------------------------------------------------------


class TestHistoryRecord:
    """Tests for HistoryRecord model."""

    def test_basic_construction(self):
        record = HistoryRecord(
            date="2024-01-15",
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000,
        )
        assert record.date == "2024-01-15"
        assert record.close == 153.0
        assert record.volume == 1000000

    def test_model_dump(self):
        record = HistoryRecord(
            date="2024-01-15", open=150.0, high=155.0,
            low=149.0, close=153.0, volume=1000000,
        )
        d = record.model_dump()
        assert d["date"] == "2024-01-15"
        assert isinstance(d, dict)


class TestTickerData:
    """Tests for TickerData model with alias handling."""

    def test_basic_construction(self):
        data = TickerData(ticker="AAPL", name="Apple Inc.")
        assert data.ticker == "AAPL"
        assert data.current_price is None
        assert data.recent_history == []

    def test_alias_roundtrip(self):
        """w52_high/w52_low fields serialize to 52w_high/52w_low via alias."""
        data = TickerData(
            ticker="AAPL",
            name="Apple Inc.",
            w52_high=180.0,
            w52_low=120.0,
        )
        d = data.model_dump(by_alias=True)
        assert d["52w_high"] == 180.0
        assert d["52w_low"] == 120.0
        assert "w52_high" not in d

    def test_construct_from_alias(self):
        """TickerData can be constructed using the alias key names."""
        data = TickerData(**{"ticker": "AAPL", "name": "Apple", "52w_high": 180.0})
        assert data.w52_high == 180.0

    def test_full_construction(self):
        data = TickerData(
            ticker="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            market_cap=2_500_000_000_000,
            pe_ratio=28.5,
            w52_high=180.0,
            w52_low=120.0,
            sector="Technology",
            industry="Consumer Electronics",
            recent_history=[
                HistoryRecord(
                    date="2024-01-15", open=150.0, high=155.0,
                    low=149.0, close=153.0, volume=1000000,
                ),
            ],
        )
        d = data.model_dump(by_alias=True)
        assert d["ticker"] == "AAPL"
        assert d["current_price"] == 150.0
        assert len(d["recent_history"]) == 1
        assert d["recent_history"][0]["date"] == "2024-01-15"


# ---------------------------------------------------------------------------
# Knowledge Base Model
# ---------------------------------------------------------------------------


class TestKnowledgeResult:
    """Tests for KnowledgeResult model."""

    def test_defaults(self):
        result = KnowledgeResult(text="Some knowledge")
        assert result.source == "local"
        assert result.title == ""

    def test_with_url(self):
        result = KnowledgeResult(
            text="ETF Guide",
            source="https://example.com/etf",
            title="ETF Basics",
        )
        d = result.model_dump()
        assert d["source"] == "https://example.com/etf"
        assert d["title"] == "ETF Basics"


# ---------------------------------------------------------------------------
# News Search Model
# ---------------------------------------------------------------------------


class TestNewsResult:
    """Tests for NewsResult model."""

    def test_defaults(self):
        result = NewsResult()
        assert result.title == "No title"
        assert result.description == "No description"
        assert result.url == ""
        assert result.age == ""

    def test_from_api_data(self):
        result = NewsResult(
            title="Apple Earnings",
            description="Beats estimates",
            url="https://example.com/apple",
            age="2d",
        )
        assert result.title == "Apple Earnings"
        assert result.age == "2d"


# ---------------------------------------------------------------------------
# ParseResultModel
# ---------------------------------------------------------------------------


class TestParseResultModel:
    """Tests for ParseResultModel (validation companion)."""

    def test_defaults(self):
        """All fields have sensible defaults."""
        model = ParseResultModel()
        d = model.model_dump()
        assert d["question_type"] == "knowledge"
        assert d["tickers"] == []
        assert d["company_names"] == []
        assert d["time_period"] is None
        assert d["needs_news"] is False
        assert d["knowledge_query"] is None

    def test_question_type_analysis(self):
        """question_type 'analysis' is accepted."""
        model = ParseResultModel(question_type="analysis")
        assert model.question_type == "analysis"

    def test_question_type_knowledge(self):
        """question_type 'knowledge' is accepted."""
        model = ParseResultModel(question_type="knowledge")
        assert model.question_type == "knowledge"

    def test_invalid_question_type_coerced_to_knowledge(self):
        """Invalid question_type values are coerced to 'knowledge' (safe default)."""
        for invalid in ["data", "mixed", "news", "", "ANALYSIS"]:
            model = ParseResultModel(question_type=invalid)
            assert model.question_type == "knowledge", (
                f"Expected 'knowledge' for {invalid!r}, got {model.question_type!r}"
            )

    def test_validates_llm_output(self):
        """Typical LLM JSON output is validated correctly."""
        llm_output = {
            "tickers": ["AAPL"],
            "company_names": ["Apple"],
            "time_period": "5d",
            "time_start": None,
            "time_end": None,
            "asset_type": "equity",
            "sector": None,
            "needs_news": True,
            "news_query": "Apple earnings",
            "knowledge_query": None,
        }
        model = ParseResultModel(**llm_output)
        assert model.tickers == ["AAPL"]
        assert model.needs_news is True
        assert model.news_query == "Apple earnings"

    def test_extra_fields_ignored(self):
        """Extra fields from LLM are silently dropped."""
        data = {"tickers": [], "unexpected_field": "ignored"}
        model = ParseResultModel(**data)
        assert model.tickers == []
        assert not hasattr(model, "unexpected_field")

    def test_partial_output_gets_defaults(self):
        """Missing fields get defaults (handles incomplete LLM output)."""
        data = {"tickers": ["TSLA"]}
        model = ParseResultModel(**data)
        assert model.tickers == ["TSLA"]
        assert model.needs_news is False
        assert model.knowledge_query is None

    def test_valid_time_periods_accepted(self):
        """All valid yfinance period strings are accepted."""
        valid = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        for period in valid:
            model = ParseResultModel(time_period=period)
            assert model.time_period == period

    def test_invalid_time_period_coerced_to_none(self):
        """Invalid time_period values (e.g. '2w') are coerced to None."""
        invalid = ["2w", "4mo", "2d", "3y", "1w", "15d", "half_month"]
        for period in invalid:
            model = ParseResultModel(time_period=period)
            assert model.time_period is None, f"Expected None for {period!r}, got {model.time_period!r}"

    def test_none_time_period_stays_none(self):
        """None time_period is preserved."""
        model = ParseResultModel(time_period=None)
        assert model.time_period is None

    def test_valid_date_formats_accepted(self):
        """YYYY-MM-DD date strings are accepted for time_start/time_end."""
        model = ParseResultModel(time_start="2026-01-15", time_end="2026-01-16")
        assert model.time_start == "2026-01-15"
        assert model.time_end == "2026-01-16"

    def test_invalid_date_formats_coerced_to_none(self):
        """Non-YYYY-MM-DD date strings are coerced to None."""
        invalid_dates = [
            "01/15/2026",        # MM/DD/YYYY
            "January 15, 2026",  # human readable
            "2026/01/15",        # YYYY/MM/DD
            "15-01-2026",        # DD-MM-YYYY
            "1/15",              # M/DD (no year)
            "2026-1-15",         # single-digit month
            "not-a-date",
        ]
        for bad_date in invalid_dates:
            model = ParseResultModel(time_start=bad_date)
            assert model.time_start is None, f"Expected None for {bad_date!r}, got {model.time_start!r}"

    def test_none_dates_stay_none(self):
        """None time_start/time_end are preserved."""
        model = ParseResultModel(time_start=None, time_end=None)
        assert model.time_start is None
        assert model.time_end is None

    def test_time_period_cleared_when_dates_set(self):
        """time_period is forced to None when time_start is set (date range takes priority)."""
        model = ParseResultModel(
            time_period="1d",
            time_start="2026-01-15",
            time_end="2026-01-16",
        )
        assert model.time_period is None
        assert model.time_start == "2026-01-15"
        assert model.time_end == "2026-01-16"

    def test_time_period_preserved_without_dates(self):
        """time_period is preserved when no time_start is set."""
        model = ParseResultModel(time_period="5d", time_start=None)
        assert model.time_period == "5d"
        assert model.time_start is None


# ---------------------------------------------------------------------------
# Fundamental Data Models
# ---------------------------------------------------------------------------


class TestFundamentalDataModels:
    """Tests for FMP fundamental data Pydantic models."""

    def test_income_statement_record(self):
        record = IncomeStatementRecord(
            date="2025-09-30", period="FY", revenue=394328000000
        )
        assert record.revenue == 394328000000
        assert record.eps is None  # default

    def test_fundamental_data_defaults(self):
        data = FundamentalData(ticker="AAPL", name="Apple Inc.")
        assert data.income_statements == []
        assert data.transcript_summary == ""

    def test_needs_fundamentals_default(self):
        model = ParseResultModel()
        assert model.needs_fundamentals is False
        assert model.fundamental_endpoints == []

    def test_valid_fundamental_endpoints(self):
        model = ParseResultModel(
            fundamental_endpoints=["financial_statement", "earnings_transcript"]
        )
        assert model.fundamental_endpoints == ["financial_statement", "earnings_transcript"]

    def test_invalid_fundamental_endpoints_filtered(self):
        model = ParseResultModel(
            fundamental_endpoints=["financial_statement", "invalid_endpoint", "income_statement"]
        )
        # Only "financial_statement" is valid; "income_statement" is now invalid (old granular name)
        assert model.fundamental_endpoints == ["financial_statement"]

    def test_earnings_transcript_endpoint_accepted(self):
        model = ParseResultModel(
            fundamental_endpoints=["earnings_transcript"]
        )
        assert model.fundamental_endpoints == ["earnings_transcript"]


# ---------------------------------------------------------------------------
# Trace Event Models
# ---------------------------------------------------------------------------


class TestTraceEvents:
    """Tests for trace event models."""

    def test_trace_event(self):
        event = TraceEvent(stage="parse", status="started", detail="Parsing...")
        d = event.model_dump()
        assert d["event_type"] == "trace"
        assert d["stage"] == "parse"
        assert d["status"] == "started"

    def test_trace_event_with_tools(self):
        event = TraceEvent(
            stage="route", status="completed",
            detail="Routing", tools=["fetch_market_data"],
        )
        d = event.model_dump()
        assert d["tools"] == ["fetch_market_data"]

    def test_tool_input_event(self):
        event = ToolInputEvent(tool="market_data", input={"tickers": ["AAPL"]})
        d = event.model_dump()
        assert d["event_type"] == "tool_input"
        assert d["tool"] == "market_data"
        assert d["input"]["tickers"] == ["AAPL"]

    def test_tool_output_event(self):
        event = ToolOutputEvent(tool="market_data", output="AAPL: $150")
        d = event.model_dump()
        assert d["event_type"] == "tool_output"
        assert d["output"] == "AAPL: $150"

    def test_answer_event(self):
        event = AnswerEvent(answer="The stock is at $150.")
        d = event.model_dump()
        assert d["event_type"] == "answer"
        assert d["answer"] == "The stock is at $150."

    def test_error_event(self):
        event = ErrorEvent(message="Something went wrong")
        d = event.model_dump()
        assert d["event_type"] == "error"
        assert d["message"] == "Something went wrong"
