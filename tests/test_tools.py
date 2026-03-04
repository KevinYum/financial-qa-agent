"""Unit tests for individual tool modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Market Data Tool — Ticker Extraction (regex fallback)
# ---------------------------------------------------------------------------


class TestExtractTickers:
    """Tests for ticker extraction from question text."""

    def test_dollar_sign_tickers(self):
        from src.financial_qa_agent.tools.market_data import _extract_tickers

        result = _extract_tickers("Compare $TSLA and $GOOG")
        assert "TSLA" in result
        assert "GOOG" in result

    def test_uppercase_tickers(self):
        from src.financial_qa_agent.tools.market_data import _extract_tickers

        result = _extract_tickers("What is the price of AAPL?")
        assert "AAPL" in result

    def test_filters_common_words(self):
        from src.financial_qa_agent.tools.market_data import _extract_tickers

        result = _extract_tickers("What is inflation?")
        assert result == []

    def test_deduplicates(self):
        from src.financial_qa_agent.tools.market_data import _extract_tickers

        result = _extract_tickers("$AAPL and AAPL stock")
        assert result.count("AAPL") == 1

    def test_no_tickers(self):
        from src.financial_qa_agent.tools.market_data import _extract_tickers

        result = _extract_tickers("How does compound interest work?")
        assert result == []


# ---------------------------------------------------------------------------
# Market Data Tool — fetch_market_data (no entities / regex fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_data_no_tickers():
    """No tickers found returns appropriate message."""
    from src.financial_qa_agent.tools.market_data import fetch_market_data

    result = await fetch_market_data("What is inflation?")
    assert "No stock tickers" in result


@pytest.mark.asyncio
async def test_market_data_fetch_with_mock():
    """Market data fetch with mocked yfinance (regex fallback path)."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Apple Inc.",
        "currentPrice": 150.0,
        "marketCap": 2_500_000_000_000,
        "trailingPE": 28.5,
        "fiftyTwoWeekHigh": 180.0,
        "fiftyTwoWeekLow": 120.0,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ):
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        result = await fetch_market_data("Price of AAPL?")
        assert "Apple Inc." in result
        assert "150" in result
        assert "Technology" in result


# ---------------------------------------------------------------------------
# Market Data Tool — Parse-result-driven behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_data_entities_tickers():
    """Parse result tickers are used instead of regex extraction."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Tesla Inc.", "currentPrice": 250.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_yf:
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        # Question has no uppercase ticker, but parse_result resolves "Tesla" → TSLA
        result = await fetch_market_data(
            "How is Tesla doing?",
            parse_result={"tickers": ["TSLA"], "company_names": ["Tesla"]},
        )
        mock_yf.assert_called_with("TSLA")
        assert "Tesla Inc." in result


@pytest.mark.asyncio
async def test_market_data_entities_time_period():
    """Parse result time_period is forwarded to yfinance history()."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currentPrice": 150.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ):
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        await fetch_market_data(
            "Apple stock last 3 months",
            parse_result={"tickers": ["AAPL"], "time_period": "3mo"},
        )
        mock_ticker.history.assert_called_with(period="3mo")


@pytest.mark.asyncio
async def test_market_data_entities_date_range():
    """Parse result time_start/time_end use date range instead of period."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currentPrice": 150.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ):
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        await fetch_market_data(
            "AAPL from Jan to June 2024",
            parse_result={
                "tickers": ["AAPL"],
                "time_start": "2024-01-01",
                "time_end": "2024-06-30",
            },
        )
        mock_ticker.history.assert_called_with(
            start="2024-01-01", end="2024-06-30"
        )


@pytest.mark.asyncio
async def test_market_data_sector_fallback():
    """Sector queries resolve to sector ETF when no tickers provided."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Technology Select Sector SPDR Fund",
        "currentPrice": 200.0,
    }
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_yf:
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        result = await fetch_market_data(
            "How is the tech sector doing?",
            parse_result={"asset_type": "sector", "sector": "technology"},
        )
        mock_yf.assert_called_with("XLK")
        assert "Technology Select Sector" in result


@pytest.mark.asyncio
async def test_market_data_crypto_symbol():
    """Crypto symbols (BTC-USD) are passed through from parse result."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Bitcoin USD", "currentPrice": 60000.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_yf:
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        result = await fetch_market_data(
            "Bitcoin price",
            parse_result={"tickers": ["BTC-USD"], "asset_type": "crypto"},
        )
        mock_yf.assert_called_with("BTC-USD")
        assert "Bitcoin USD" in result


@pytest.mark.asyncio
async def test_market_data_entities_empty_falls_back_to_regex():
    """Empty parse_result dict falls back to regex ticker extraction."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currentPrice": 150.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ) as mock_yf:
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        result = await fetch_market_data("Price of AAPL?", parse_result={})
        mock_yf.assert_called_with("AAPL")
        assert "Apple Inc." in result


@pytest.mark.asyncio
async def test_market_data_invalid_period_defaults():
    """Invalid time_period in parse result defaults to 5d."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Apple Inc.", "currentPrice": 150.0}
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ):
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        await fetch_market_data(
            "AAPL last 2 months",
            parse_result={"tickers": ["AAPL"], "time_period": "2mo"},
        )
        # "2mo" is not valid for yfinance, should default to "5d"
        mock_ticker.history.assert_called_with(period="5d")


# ---------------------------------------------------------------------------
# News Search Tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_news_search_no_api_key():
    """Missing API key returns graceful message."""
    with patch("src.financial_qa_agent.tools.news_search.settings") as mock_settings:
        mock_settings.brave_api_key = ""
        from src.financial_qa_agent.tools.news_search import fetch_news

        result = await fetch_news("Latest AAPL earnings")
        assert "not configured" in result


@pytest.mark.asyncio
async def test_news_search_with_results():
    """News search returns formatted results from Brave API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Apple Reports Q4 Earnings",
                    "description": "Apple beats expectations",
                    "url": "https://example.com/apple-q4",
                    "age": "2d",
                },
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.financial_qa_agent.tools.news_search.settings") as mock_settings,
        patch("src.financial_qa_agent.tools.news_search.httpx.AsyncClient", return_value=mock_client),
    ):
        mock_settings.brave_api_key = "test-key"
        from src.financial_qa_agent.tools.news_search import fetch_news

        result = await fetch_news("AAPL earnings")
        assert "Apple Reports Q4 Earnings" in result
        assert "beats expectations" in result


@pytest.mark.asyncio
async def test_news_search_uses_refined_query():
    """News search uses parse_result.news_query instead of raw question."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {"results": [{"title": "Tesla News", "description": "Latest", "url": "https://x.com", "age": "1d"}]}
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.financial_qa_agent.tools.news_search.settings") as mock_settings,
        patch("src.financial_qa_agent.tools.news_search.httpx.AsyncClient", return_value=mock_client),
    ):
        mock_settings.brave_api_key = "test-key"
        from src.financial_qa_agent.tools.news_search import fetch_news

        await fetch_news(
            "What's the latest news on Tesla and how does it compare to last year?",
            parse_result={"news_query": "Tesla latest news"},
        )
        # Verify the refined query was used, not the raw question
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["q"] == "Tesla latest news"


# ---------------------------------------------------------------------------
# Knowledge Base Tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_base_good_results():
    """Knowledge base returns good ChromaDB results without fallback."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_kb", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["doc1", "doc2"],
        documents=[
            "Compound interest is interest calculated on both the initial "
            "principal and the accumulated interest from previous periods.",
            "Simple interest is calculated only on the original principal amount.",
        ],
        metadatas=[{"source": "local"}, {"source": "local"}],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ):
        from src.financial_qa_agent.tools.knowledge_base import fetch_knowledge

        result = await fetch_knowledge("What is compound interest?")
        assert "interest" in result.lower()


@pytest.mark.asyncio
async def test_knowledge_base_fallback_to_web():
    """Knowledge base falls back to Brave when ChromaDB results are sparse."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_kb_empty", metadata={"hnsw:space": "cosine"}
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Options Trading Guide",
                    "description": "Learn about options",
                    "url": "https://example.com/options",
                },
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_response
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.financial_qa_agent.tools.knowledge_base._get_collection",
            return_value=collection,
        ),
        patch(
            "src.financial_qa_agent.tools.knowledge_base.settings"
        ) as mock_settings,
        patch(
            "src.financial_qa_agent.tools.knowledge_base.httpx.AsyncClient",
            return_value=mock_http_client,
        ),
    ):
        mock_settings.brave_api_key = "test-key"
        mock_settings.kb_min_results = 2
        mock_settings.kb_max_distance = 0.5
        from src.financial_qa_agent.tools.knowledge_base import fetch_knowledge

        result = await fetch_knowledge("What are options?")
        assert "Options Trading Guide" in result


@pytest.mark.asyncio
async def test_knowledge_base_fallback_uses_refined_query():
    """Knowledge base fallback uses parse_result.news_query for Brave search."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_kb_refined", metadata={"hnsw:space": "cosine"}
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {"results": [{"title": "Bonds 101", "description": "Guide", "url": "https://x.com"}]}
    }
    mock_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.get.return_value = mock_response
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.financial_qa_agent.tools.knowledge_base._get_collection",
            return_value=collection,
        ),
        patch(
            "src.financial_qa_agent.tools.knowledge_base.settings"
        ) as mock_settings,
        patch(
            "src.financial_qa_agent.tools.knowledge_base.httpx.AsyncClient",
            return_value=mock_http_client,
        ),
    ):
        mock_settings.brave_api_key = "test-key"
        mock_settings.kb_min_results = 2
        mock_settings.kb_max_distance = 0.5
        from src.financial_qa_agent.tools.knowledge_base import fetch_knowledge

        await fetch_knowledge(
            "Explain how bonds work and their relationship to interest rates",
            parse_result={"news_query": "bonds interest rates explained"},
        )
        # Verify the refined query was used for Brave fallback
        call_args = mock_http_client.get.call_args
        assert call_args[1]["params"]["q"] == "bonds interest rates explained"


@pytest.mark.asyncio
async def test_knowledge_base_uses_knowledge_queries():
    """Knowledge base uses knowledge_queries from parse result for ChromaDB search."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_kb_queries", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["doc1"],
        documents=[
            "A stock option gives the holder the right but not the obligation "
            "to buy or sell a stock at a predetermined price."
        ],
        metadatas=[{"source": "local"}],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ):
        from src.financial_qa_agent.tools.knowledge_base import fetch_knowledge

        result = await fetch_knowledge(
            "How do AAPL stock options work and should I exercise them?",
            parse_result={
                "knowledge_queries": [
                    "What are stock options?",
                    "When should you exercise stock options?",
                ],
            },
        )
        assert "option" in result.lower()
