"""Fundamental data tool — fetches financial statements and earnings transcripts via FMP API.

Two FMP API calls:
- ``financial_statement`` → income statement via ``/income-statement``
- ``earnings_transcript`` → earnings call transcript via ``/earning-call-transcript``

Only works for equities. Non-equity tickers (crypto, commodities, forex, indices)
are gracefully rejected.
"""

import logging
from datetime import date, timedelta

import httpx
from langchain_core.messages import HumanMessage

from ..config import settings
from ..models import FundamentalData, IncomeStatementRecord

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Tickers with these suffixes/prefixes are NOT equities
_NON_EQUITY_SUFFIXES = ("-USD", "=X", "=F")
_NON_EQUITY_PREFIXES = ("^",)

# Asset types that support FMP fundamental data
_EQUITY_ASSET_TYPES = frozenset({"equity", None})

SUMMARIZE_TRANSCRIPT_PROMPT = """\
You are a financial analyst assistant. Summarize the following earnings call \
transcript, focusing on information relevant to the user's question.

User Question: {question}

Extract and organize the key points into these categories (skip any category \
that has no relevant content):
- **Financial Highlights**: Revenue, earnings, margins, growth numbers mentioned
- **Guidance & Outlook**: Forward-looking statements, forecasts, targets
- **Strategic Updates**: New products, partnerships, market expansion, restructuring
- **Risk Factors**: Challenges, headwinds, concerns raised by management or analysts
- **Key Q&A Points**: Notable analyst questions and management responses

Be concise — aim for 300-500 words total. Include specific numbers and quotes \
where available.

--- Transcript ---
{transcript}
"""


# ---------------------------------------------------------------------------
# FMP API helpers
# ---------------------------------------------------------------------------


async def _fetch_fmp_json(path: str, params: dict | None = None) -> list[dict]:
    """Fetch a single FMP endpoint. Returns list of dicts or empty list."""
    url = f"{FMP_BASE_URL}{path}"
    query = {"apikey": settings.fmp_api_key}
    if params:
        query.update(params)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=query)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, list):
        return data
    # Some FMP errors come back as dicts: {"Error Message": "..."}
    if isinstance(data, dict) and "Error Message" in data:
        logger.warning("FMP API error: %s", data["Error Message"])
        return []
    return []


# ---------------------------------------------------------------------------
# Mapper: FMP JSON → Pydantic model
# ---------------------------------------------------------------------------


def _map_income_statement(raw: dict) -> IncomeStatementRecord:
    return IncomeStatementRecord(
        date=raw.get("date", ""),
        period=raw.get("period", ""),
        revenue=raw.get("revenue"),
        gross_profit=raw.get("grossProfit"),
        operating_income=raw.get("operatingIncome"),
        net_income=raw.get("netIncome"),
        eps=raw.get("eps"),
        ebitda=raw.get("ebitda"),
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_dollar(v: float) -> str:
    """Format a dollar value with commas."""
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def _format_fundamental_data(data: FundamentalData) -> str:
    """Format fundamental data into concise text for LLM consumption."""
    lines = [f"=== {data.name} ({data.ticker}) — Fundamental Data ==="]

    if data.income_statements:
        lines.append("\n--- Income Statement ---")
        for stmt in data.income_statements:
            lines.append(f"  {stmt.date} ({stmt.period}):")
            if stmt.revenue is not None:
                lines.append(f"    Revenue: {_fmt_dollar(stmt.revenue)}")
            if stmt.gross_profit is not None:
                lines.append(f"    Gross Profit: {_fmt_dollar(stmt.gross_profit)}")
            if stmt.operating_income is not None:
                lines.append(f"    Operating Income: {_fmt_dollar(stmt.operating_income)}")
            if stmt.net_income is not None:
                lines.append(f"    Net Income: {_fmt_dollar(stmt.net_income)}")
            if stmt.eps is not None:
                lines.append(f"    EPS: ${stmt.eps:.2f}")
            if stmt.ebitda is not None:
                lines.append(f"    EBITDA: {_fmt_dollar(stmt.ebitda)}")

    if data.transcript_summary:
        lines.append("\n--- Earnings Call Transcript (Summary) ---")
        lines.append(data.transcript_summary)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Period / quarter resolution
# ---------------------------------------------------------------------------


def _resolve_period_and_limit(parse_result: dict) -> tuple[str, int]:
    """Determine annual vs quarterly and record limit from parse_result."""
    time_period = parse_result.get("time_period")
    # Short periods suggest quarterly data
    if time_period in ("1d", "5d", "1mo", "3mo", "6mo"):
        return "quarter", 4
    return "annual", 4


def _resolve_transcript_quarter_year(parse_result: dict) -> tuple[int, int]:
    """Determine which quarter/year to fetch for earnings transcript.

    Defaults to the most recently completed quarter relative to today.
    """
    # If user specified dates, try to infer from time_end or time_start
    time_end = parse_result.get("time_end") or parse_result.get("time_start")
    if time_end:
        try:
            parts = time_end.split("-")
            y, m = int(parts[0]), int(parts[1])
            q = (m - 1) // 3 + 1
            return q, y
        except (ValueError, IndexError):
            pass

    # Default: most recently completed quarter
    today = date.today()
    # Go back ~45 days to be safe (transcripts take a few weeks to appear)
    ref = today - timedelta(days=45)
    q = (ref.month - 1) // 3 + 1
    return q, ref.year


def _is_equity_ticker(ticker: str) -> bool:
    """Check if a ticker looks like an equity (not crypto/commodity/forex/index)."""
    if any(ticker.startswith(p) for p in _NON_EQUITY_PREFIXES):
        return False
    if any(ticker.endswith(s) for s in _NON_EQUITY_SUFFIXES):
        return False
    return True


# ---------------------------------------------------------------------------
# Transcript summarization
# ---------------------------------------------------------------------------


async def _summarize_transcript(transcript: str, question: str) -> str:
    """Use LLM to summarize an earnings call transcript."""
    # Import here to avoid circular import at module level
    from ..agent import _build_llm

    llm = _build_llm()
    prompt = SUMMARIZE_TRANSCRIPT_PROMPT.format(
        question=question,
        transcript=transcript[:30000],  # Safety cap on transcript length
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_fundamental_data(
    question: str, parse_result: dict | None = None
) -> str:
    """Fetch fundamental data from FMP — two API calls max per ticker.

    Data sources (controlled by ``parse_result.fundamental_endpoints``):
    - ``financial_statement`` → FMP ``/income-statement`` (1 API call)
    - ``earnings_transcript`` → FMP ``/earning-call-transcript`` (1 API call)

    Only works for equities. Returns graceful messages for:
    - Missing API key
    - Non-equity asset types (crypto, commodities, forex, indices)
    - No tickers in parse result
    """
    if not settings.fmp_api_key:
        return "Fundamental data is not configured (missing FMP_API_KEY)."

    parse_result = parse_result or {}
    tickers = list(parse_result.get("tickers") or [])

    if not tickers:
        return "No stock tickers identified for fundamental data lookup."

    # Check asset_type guard
    asset_type = parse_result.get("asset_type")
    if asset_type and asset_type not in _EQUITY_ASSET_TYPES:
        return (
            f"Fundamental data is only available for equities "
            f"(requested asset type: {asset_type})."
        )

    # Filter to equity tickers only
    equity_tickers = [t for t in tickers if _is_equity_ticker(t)]
    if not equity_tickers:
        return (
            "Fundamental data is only available for equities. "
            "The identified tickers are non-equity instruments."
        )

    endpoints = parse_result.get("fundamental_endpoints") or []
    if not endpoints:
        return "No fundamental data endpoints requested by parser."

    period, limit = _resolve_period_and_limit(parse_result)

    results: list[str] = []

    for ticker in equity_tickers[:3]:  # Cap at 3 tickers for API budget
        fundamental = FundamentalData(ticker=ticker, name=ticker)
        fetch_errors: list[str] = []

        # --- Financial statement (income statement only — 1 FMP call) ---
        if "financial_statement" in endpoints:
            try:
                raw_data = await _fetch_fmp_json(
                    "/income-statement",
                    {"symbol": ticker, "period": period, "limit": str(limit)},
                )
                if raw_data:
                    fundamental.income_statements = [
                        _map_income_statement(r) for r in raw_data[:limit]
                    ]
                else:
                    fetch_errors.append("financial_statement: no data returned")
            except Exception as exc:
                logger.warning("FMP income statement failed for %s: %s", ticker, exc)
                fetch_errors.append(f"financial_statement: {exc}")

        # --- Earnings transcript (1 FMP call) ---
        if "earnings_transcript" in endpoints:
            quarter, year = _resolve_transcript_quarter_year(parse_result)
            try:
                raw_data = await _fetch_fmp_json(
                    "/earning-call-transcript",
                    {"symbol": ticker, "quarter": str(quarter), "year": str(year)},
                )
                if raw_data and raw_data[0].get("content"):
                    transcript_text = raw_data[0]["content"]
                    fundamental.transcript_summary = await _summarize_transcript(
                        transcript_text, question
                    )
                else:
                    fetch_errors.append(f"earnings_transcript Q{quarter} {year}: no data returned")
            except Exception as exc:
                logger.warning(
                    "FMP transcript fetch failed for %s Q%d %d: %s",
                    ticker, quarter, year, exc,
                )
                fetch_errors.append(f"earnings_transcript: {exc}")

        formatted = _format_fundamental_data(fundamental)
        if fetch_errors:
            formatted += "\n\n[Note: Some data could not be retrieved: " + "; ".join(fetch_errors) + "]"
        results.append(formatted)

    return "\n\n".join(results) if results else "No fundamental data retrieved."
