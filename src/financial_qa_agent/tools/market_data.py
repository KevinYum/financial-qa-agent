"""Market data tool — fetches OHLCV and fundamentals via yfinance."""

import asyncio
import logging
from functools import partial

import yfinance as yf

from ..models import HistoryRecord, TickerData

logger = logging.getLogger(__name__)

# Valid yfinance period strings
VALID_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})


def _fetch_ticker_data(
    ticker: str,
    period: str = "5d",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Synchronous yfinance fetch for a single ticker.

    If start is provided, uses date range (start/end). Otherwise uses period.

    All data points are passed to the LLM — the synthesize prompt instructs
    it to select only question-relevant facts for the response.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    if start:
        hist = t.history(start=start, end=end)
    else:
        hist = t.history(period=period)

    history_records: list[HistoryRecord] = []
    if not hist.empty:
        for date, row in hist.iterrows():
            history_records.append(HistoryRecord(
                date=str(date.date()),
                open=round(row.get("Open", 0), 2),
                high=round(row.get("High", 0), 2),
                low=round(row.get("Low", 0), 2),
                close=round(row.get("Close", 0), 2),
                volume=int(row.get("Volume", 0)),
            ))

    ticker_data = TickerData(
        ticker=ticker,
        name=info.get("longName", ticker),
        current_price=info.get("currentPrice"),
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        w52_high=info.get("fiftyTwoWeekHigh"),
        w52_low=info.get("fiftyTwoWeekLow"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        recent_history=history_records,
    )
    return ticker_data.model_dump(by_alias=True)


def _format_ticker_data(data: dict) -> str:
    """Format ticker data dict into readable text for LLM consumption."""
    lines = [f"=== {data['name']} ({data['ticker']}) ==="]
    if data["current_price"]:
        lines.append(f"Current Price: ${data['current_price']}")
    if data["market_cap"]:
        lines.append(f"Market Cap: ${data['market_cap']:,.0f}")
    if data["pe_ratio"]:
        lines.append(f"P/E Ratio: {data['pe_ratio']:.2f}")
    if data["52w_high"] and data["52w_low"]:
        lines.append(f"52-Week Range: ${data['52w_low']} - ${data['52w_high']}")
    if data["sector"]:
        lines.append(f"Sector: {data['sector']} / {data.get('industry', 'N/A')}")
    if data["recent_history"]:
        lines.append("OHLCV:")
        for h in data["recent_history"]:
            lines.append(
                f"  {h['date']}: O={h['open']} H={h['high']} "
                f"L={h['low']} C={h['close']} V={h['volume']:,}"
            )
    return "\n".join(lines)


async def fetch_market_data(question: str, parse_result: dict | None = None) -> str:
    """Fetch market data for instruments identified by the parse node.

    Tickers come exclusively from parse_result (LLM extraction).
    The parse LLM handles all resolution — company names to tickers,
    sectors to ETFs, crypto/forex/index symbols.

    Time period comes from parse_result, defaults to "5d".
    """
    parse_result = parse_result or {}

    # Tickers must come from LLM parse result — no fallback
    tickers = list(parse_result.get("tickers") or [])

    if not tickers:
        logger.debug("No tickers in parse result for question: %r", question)
        return "No stock tickers or financial instruments identified in the question."

    logger.debug("Resolved tickers: %s", tickers[:5])

    # 2. Resolve time parameters
    period = parse_result.get("time_period") or "5d"
    if period not in VALID_PERIODS:
        logger.debug("Invalid period %r, defaulting to 5d", period)
        period = "5d"
    start = parse_result.get("time_start")
    end = parse_result.get("time_end")

    # 3. Fetch data
    loop = asyncio.get_event_loop()
    results: list[dict] = []
    for ticker in tickers[:5]:  # Cap at 5 tickers for comparison queries
        logger.debug("Fetching yfinance data: ticker=%s period=%s start=%s end=%s",
                      ticker, period, start, end)
        data = await loop.run_in_executor(
            None, partial(_fetch_ticker_data, ticker, period, start, end)
        )
        logger.debug("Fetched %s: price=%s", ticker, data.get("current_price"))
        results.append(data)

    parts = [_format_ticker_data(r) for r in results]
    return "\n\n".join(parts)
