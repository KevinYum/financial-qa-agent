"""Market data tool — fetches OHLCV and fundamentals via yfinance."""

import asyncio
import re
from functools import partial

import yfinance as yf

# Common English words that look like tickers but aren't
_STOP_WORDS = frozenset({
    "A", "I", "AM", "IS", "IT", "AT", "IN", "ON", "TO", "THE", "AND", "OR",
    "FOR", "OF", "BY", "AN", "AS", "IF", "DO", "SO", "UP", "MY", "ME", "WE",
    "US", "BE", "NO", "NOT", "BUT", "HOW", "HAS", "HAD", "ARE", "WAS", "ALL",
    "CAN", "HER", "HIS", "ITS", "MAY", "NEW", "NOW", "OLD", "OUR", "OUT",
    "OWN", "SAY", "SHE", "TOO", "USE", "WAY", "WHO", "DID", "GET", "HIM",
    "LET", "PUT", "RUN", "SET", "TOP", "WHY", "BIG", "END", "DAY", "GOT",
    "LOW", "HIGH", "VS", "ETF", "IPO", "CEO", "CFO", "GDP", "SEC", "FED",
    "NYSE", "WHAT", "WHEN", "WILL", "WITH", "THAT", "THIS", "FROM", "HAVE",
    "BEEN", "THAN", "THEM", "THEN", "THEY", "ABOUT", "AFTER", "COULD", "WOULD",
    "SHOULD", "THEIR", "THERE", "THESE", "THOSE", "WHERE", "WHICH", "WHILE",
    "STOCK", "PRICE", "MUCH", "MANY", "SOME",
})

# Valid yfinance period strings
VALID_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})

# SPDR sector ETF mapping (fallback when LLM doesn't resolve to tickers)
SECTOR_ETFS: dict[str, str] = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication_services": "XLC",
}


def _extract_tickers(question: str) -> list[str]:
    """Extract stock ticker symbols from a question string (regex fallback).

    Looks for $TICKER patterns and standalone uppercase 1-5 letter words.
    Filters out common English words.
    """
    # Match $TICKER pattern
    dollar_tickers = re.findall(r"\$([A-Z]{1,5})", question.upper())
    # Match standalone uppercase words 1-5 chars in original text
    word_tickers = re.findall(r"\b([A-Z]{1,5})\b", question)
    # Deduplicate, preserve order, filter stop words
    seen: set[str] = set()
    result: list[str] = []
    for t in dollar_tickers + word_tickers:
        if t not in seen and t not in _STOP_WORDS:
            seen.add(t)
            result.append(t)
    return result


def _fetch_ticker_data(
    ticker: str,
    period: str = "5d",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Synchronous yfinance fetch for a single ticker.

    If start is provided, uses date range (start/end). Otherwise uses period.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    if start:
        hist = t.history(start=start, end=end)
    else:
        hist = t.history(period=period)

    history_records: list[dict] = []
    if not hist.empty:
        # Show at most last 10 data points regardless of period
        for date, row in hist.tail(10).iterrows():
            history_records.append({
                "date": str(date.date()),
                "open": round(row.get("Open", 0), 2),
                "high": round(row.get("High", 0), 2),
                "low": round(row.get("Low", 0), 2),
                "close": round(row.get("Close", 0), 2),
                "volume": int(row.get("Volume", 0)),
            })

    return {
        "ticker": ticker,
        "name": info.get("longName", ticker),
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "recent_history": history_records,
    }


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
        lines.append("Recent OHLCV:")
        for h in data["recent_history"]:
            lines.append(
                f"  {h['date']}: O={h['open']} H={h['high']} "
                f"L={h['low']} C={h['close']} V={h['volume']:,}"
            )
    return "\n".join(lines)


async def fetch_market_data(question: str, parse_result: dict | None = None) -> str:
    """Fetch market data for instruments mentioned in the question.

    Resolution order for tickers:
    1. LLM-extracted parse result (tickers field)
    2. Regex fallback (_extract_tickers)
    3. Sector ETF lookup (if asset_type is "sector")

    Time period comes from parse_result, defaults to "5d".
    """
    parse_result = parse_result or {}

    # 1. Resolve tickers
    tickers = list(parse_result.get("tickers") or [])
    if not tickers:
        tickers = _extract_tickers(question)
    if not tickers and parse_result.get("asset_type") == "sector":
        sector = (parse_result.get("sector") or "").lower().replace(" ", "_")
        etf = SECTOR_ETFS.get(sector)
        if etf:
            tickers = [etf]

    if not tickers:
        return "No stock tickers or financial instruments identified in the question."

    # 2. Resolve time parameters
    period = parse_result.get("time_period") or "5d"
    if period not in VALID_PERIODS:
        period = "5d"
    start = parse_result.get("time_start")
    end = parse_result.get("time_end")

    # 3. Fetch data
    loop = asyncio.get_event_loop()
    results: list[dict] = []
    for ticker in tickers[:5]:  # Cap at 5 tickers for comparison queries
        data = await loop.run_in_executor(
            None, partial(_fetch_ticker_data, ticker, period, start, end)
        )
        results.append(data)

    parts = [_format_ticker_data(r) for r in results]
    return "\n\n".join(parts)
