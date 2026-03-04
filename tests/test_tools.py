"""Unit tests for individual tool modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Market Data Tool — fetch_market_data (no parse result / no tickers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_data_no_tickers():
    """No tickers found returns appropriate message."""
    from src.financial_qa_agent.tools.market_data import fetch_market_data

    result = await fetch_market_data("What is inflation?")
    assert "No stock tickers" in result


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
async def test_market_data_returns_all_data_points():
    """All OHLCV data points are passed through (no server-side truncation)."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.info = {"longName": "Alibaba", "currentPrice": 135.0}

    # Simulate 21 trading days (a full month like January)
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    data = {
        "Open": [100 + i for i in range(21)],
        "High": [102 + i for i in range(21)],
        "Low": [99 + i for i in range(21)],
        "Close": [101 + i for i in range(21)],
        "Volume": [1000000] * 21,
    }
    mock_hist = pd.DataFrame(data, index=dates)
    mock_ticker.history.return_value = mock_hist

    with patch(
        "src.financial_qa_agent.tools.market_data.yf.Ticker",
        return_value=mock_ticker,
    ):
        from src.financial_qa_agent.tools.market_data import fetch_market_data

        result = await fetch_market_data(
            "BABA January activity",
            parse_result={
                "tickers": ["BABA"],
                "time_start": "2026-01-01",
                "time_end": "2026-02-01",
            },
        )
        # All 21 data points should be present — no truncation
        assert "2026-01-02" in result  # First trading day
        assert "2026-01-30" in result  # Last trading day
        ohlcv_lines = [line for line in result.split("\n") if line.strip().startswith("2026-")]
        assert len(ohlcv_lines) == 21


@pytest.mark.asyncio
async def test_market_data_sector_via_parse():
    """Sector queries use sector ETF ticker from parse result."""
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

        # Parse LLM resolves "tech sector" → XLK ticker
        result = await fetch_market_data(
            "How is the tech sector doing?",
            parse_result={"tickers": ["XLK"], "asset_type": "sector", "sector": "technology"},
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
async def test_market_data_empty_parse_result_no_tickers():
    """Empty parse_result dict returns no-tickers message (no regex fallback)."""
    from src.financial_qa_agent.tools.market_data import fetch_market_data

    result = await fetch_market_data("Price of AAPL?", parse_result={})
    assert "No stock tickers" in result


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
# Local Knowledge Tool — fetch_local_knowledge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_knowledge_good_results():
    """Local knowledge returns good ChromaDB results."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_local_good", metadata={"hnsw:space": "cosine"})
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
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("What is compound interest?")
        assert "interest" in result.lower()


@pytest.mark.asyncio
async def test_local_knowledge_no_results():
    """Local knowledge returns empty message when ChromaDB has no matches."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_local_empty", metadata={"hnsw:space": "cosine"}
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ):
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("What are options?")
        assert "No relevant local knowledge found" in result


@pytest.mark.asyncio
async def test_local_knowledge_uses_knowledge_queries():
    """Local knowledge uses knowledge_queries from parse result for ChromaDB search."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_local_queries", metadata={"hnsw:space": "cosine"}
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
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge(
            "How do AAPL stock options work and should I exercise them?",
            parse_result={
                "knowledge_queries": [
                    "What are stock options?",
                    "When should you exercise stock options?",
                ],
            },
        )
        assert "option" in result.lower()


# ---------------------------------------------------------------------------
# Web Knowledge Tool — fetch_web_knowledge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_knowledge_success():
    """Web knowledge returns formatted results from Brave and stores in ChromaDB."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_web_success", metadata={"hnsw:space": "cosine"}
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
        from src.financial_qa_agent.tools.knowledge_base import fetch_web_knowledge

        result = await fetch_web_knowledge("What are options?")
        assert "Options Trading Guide" in result
        assert "Reference:" in result

    # Verify results were stored in ChromaDB
    stored = collection.get()
    assert len(stored["ids"]) > 0


@pytest.mark.asyncio
async def test_web_knowledge_no_api_key():
    """Web knowledge returns graceful message when no Brave API key."""
    with (
        patch(
            "src.financial_qa_agent.tools.knowledge_base._get_collection",
            return_value=MagicMock(),
        ),
        patch(
            "src.financial_qa_agent.tools.knowledge_base.settings"
        ) as mock_settings,
    ):
        mock_settings.brave_api_key = ""
        from src.financial_qa_agent.tools.knowledge_base import fetch_web_knowledge

        result = await fetch_web_knowledge("What is an ETF?")
        assert "No relevant web results found" in result


@pytest.mark.asyncio
async def test_web_knowledge_includes_reference_urls():
    """Web knowledge output includes Reference: [Title](URL) format."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_web_refs", metadata={"hnsw:space": "cosine"}
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "ETF Basics Guide",
                    "description": "A guide to exchange-traded funds.",
                    "url": "https://example.com/etf-guide",
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
        from src.financial_qa_agent.tools.knowledge_base import fetch_web_knowledge

        result = await fetch_web_knowledge("What is an ETF?")
        # Verify markdown reference format
        assert "Reference:" in result
        assert "[ETF Basics Guide](https://example.com/etf-guide)" in result
