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
    """Local knowledge returns full_text from metadata with chunk ID label."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_local_good", metadata={"hnsw:space": "cosine"})
    # Simulate post-v0.0.32 storage: document = summary, full_text + chunk_id in metadata
    collection.add(
        ids=["doc1_chunk_0"],
        documents=[
            "Compound interest: principal plus accumulated interest."
        ],
        metadatas=[{
            "source": "local",
            "full_text": "Compound interest is interest calculated on both the initial "
            "principal and the accumulated interest from previous periods. "
            "This results in exponential growth over time.",
            "chunk_id": "0",
        }],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ):
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("What is compound interest?")
        # Should return full_text, not the summary document
        assert "exponential growth" in result
        # Should include chunk ID label
        assert "(chunk 0)" in result


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
async def test_local_knowledge_uses_knowledge_query():
    """Local knowledge uses knowledge_query from parse result for ChromaDB search."""
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
                "knowledge_query": "stock option right to buy or sell",
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
        patch(
            "src.financial_qa_agent.tools.knowledge_base._summarize_for_embedding",
            new_callable=AsyncMock,
            return_value="Summary: options trading guide",
        ),
    ):
        mock_settings.brave_api_key = "test-key"
        mock_settings.kb_chunk_size = 200
        from src.financial_qa_agent.tools.knowledge_base import fetch_web_knowledge

        result = await fetch_web_knowledge("What are options?")
        assert "Options Trading Guide" in result
        assert "Reference:" in result

    # Verify results were stored in ChromaDB with full_text and chunk_id in metadata
    stored = collection.get(include=["metadatas"])
    assert len(stored["ids"]) > 0
    assert stored["metadatas"][0]["full_text"] == "Options Trading Guide: Learn about options"
    assert stored["metadatas"][0]["chunk_id"] == "0"


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
        patch(
            "src.financial_qa_agent.tools.knowledge_base._summarize_for_embedding",
            new_callable=AsyncMock,
            return_value="Summary: ETF basics guide",
        ),
    ):
        mock_settings.brave_api_key = "test-key"
        mock_settings.kb_chunk_size = 200
        from src.financial_qa_agent.tools.knowledge_base import fetch_web_knowledge

        result = await fetch_web_knowledge("What is an ETF?")
        # Verify markdown reference format
        assert "Reference:" in result
        assert "[ETF Basics Guide](https://example.com/etf-guide)" in result


# ---------------------------------------------------------------------------
# Knowledge Embedding Summarization — _summarize_for_embedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_for_embedding_short_text_passthrough():
    """Text ≤200 words is returned as-is without LLM call."""
    from src.financial_qa_agent.tools.knowledge_base import _summarize_for_embedding

    short_text = "This is a short text about compound interest."
    result = await _summarize_for_embedding(short_text)
    assert result == short_text


@pytest.mark.asyncio
async def test_summarize_for_embedding_long_text_calls_llm():
    """Text >200 words triggers LLM summarization."""
    from src.financial_qa_agent.tools.knowledge_base import _summarize_for_embedding

    long_text = " ".join(["word"] * 250)  # 250 words

    mock_llm_response = MagicMock()
    mock_llm_response.content = "A concise summary of the long text."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

    with patch(
        "src.financial_qa_agent.agent._build_llm",
        return_value=mock_llm,
    ):
        result = await _summarize_for_embedding(long_text)
        assert result == "A concise summary of the long text."
        mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_for_embedding_llm_failure_fallback():
    """LLM failure falls back to truncated prefix."""
    from src.financial_qa_agent.tools.knowledge_base import _summarize_for_embedding

    long_text = " ".join(["word"] * 250)

    with patch(
        "src.financial_qa_agent.agent._build_llm",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = await _summarize_for_embedding(long_text)
        assert len(result) <= 1000
        assert result == long_text[:1000]


@pytest.mark.asyncio
async def test_store_chromadb_saves_full_text_in_metadata():
    """_store_in_chromadb stores summary as document and full text in metadatas."""
    import chromadb
    from src.financial_qa_agent.tools.knowledge_base import _store_in_chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_store_full_text", metadata={"hnsw:space": "cosine"}
    )

    results = [
        {"text": "Full original article text about ETFs and investing.", "source": "https://x.com", "title": "ETF Guide"}
    ]

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._summarize_for_embedding",
        new_callable=AsyncMock,
        return_value="Summary: ETFs and investing.",
    ):
        await _store_in_chromadb(collection, results)

    stored = collection.get(include=["documents", "metadatas"])
    assert len(stored["ids"]) == 1
    # Document is the summary (for embedding)
    assert stored["documents"][0] == "Summary: ETFs and investing."
    # Metadata has full original text and chunk info
    assert stored["metadatas"][0]["full_text"] == "Full original article text about ETFs and investing."
    assert stored["metadatas"][0]["source"] == "https://x.com"
    assert stored["metadatas"][0]["title"] == "ETF Guide"
    assert stored["metadatas"][0]["chunk_id"] == "0"


@pytest.mark.asyncio
async def test_local_knowledge_backward_compat_no_full_text():
    """Old ChromaDB entries without full_text metadata still work (fallback to document)."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_local_backward", metadata={"hnsw:space": "cosine"}
    )
    # Pre-v0.0.31 entry: no full_text in metadata
    collection.add(
        ids=["old_doc"],
        documents=["Old document about bond yields and maturity dates."],
        metadatas=[{"source": "https://old.com", "title": "Bond Guide"}],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ):
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("What are bond yields?")
        # Should return document text as fallback
        assert "bond yields" in result.lower()


# ---------------------------------------------------------------------------
# Chunking — _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_short_passthrough():
    """Text at or under chunk_size words returns a single-element list."""
    from src.financial_qa_agent.tools.knowledge_base import _chunk_text

    text = "Short text about financial markets."
    result = _chunk_text(text, chunk_size=200)
    assert result == [text]
    assert len(result) == 1


def test_chunk_text_splits_long_text():
    """Text of 500 words with chunk_size=200 produces 3 chunks."""
    from src.financial_qa_agent.tools.knowledge_base import _chunk_text

    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    result = _chunk_text(text, chunk_size=200)
    assert len(result) == 3
    # First chunk has 200 words
    assert len(result[0].split()) == 200
    # Second chunk has 200 words
    assert len(result[1].split()) == 200
    # Third chunk has the remaining 100 words
    assert len(result[2].split()) == 100


def test_chunk_text_exact_boundary():
    """Text of exactly 400 words with chunk_size=200 produces 2 exact chunks."""
    from src.financial_qa_agent.tools.knowledge_base import _chunk_text

    words = [f"w{i}" for i in range(400)]
    text = " ".join(words)
    result = _chunk_text(text, chunk_size=200)
    assert len(result) == 2
    assert len(result[0].split()) == 200
    assert len(result[1].split()) == 200


# ---------------------------------------------------------------------------
# Chunking — _store_in_chromadb with chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_chromadb_chunks_long_text():
    """Long result (>chunk_size words) produces multiple ChromaDB entries with chunk_id metadata."""
    import chromadb
    from src.financial_qa_agent.tools.knowledge_base import _store_in_chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_store_chunks", metadata={"hnsw:space": "cosine"}
    )

    # 450 words → should produce 3 chunks (200 + 200 + 50) with kb_chunk_size=200
    long_text = " ".join([f"word{i}" for i in range(450)])
    results = [{"text": long_text, "source": "https://example.com/long", "title": "Long Article"}]

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._summarize_for_embedding",
        new_callable=AsyncMock,
        side_effect=lambda t: f"Summary of {len(t.split())} words",
    ), patch(
        "src.financial_qa_agent.tools.knowledge_base.settings"
    ) as mock_s:
        mock_s.kb_chunk_size = 200
        await _store_in_chromadb(collection, results)

    stored = collection.get(include=["documents", "metadatas"])
    assert len(stored["ids"]) == 3

    # Check each chunk has correct metadata
    for idx in range(3):
        meta = stored["metadatas"][idx]
        assert meta["chunk_id"] == str(idx)
        assert meta["source"] == "https://example.com/long"
        assert meta["title"] == "Long Article"
        # full_text is the chunk text, not the full document
        chunk_words = meta["full_text"].split()
        if idx < 2:
            assert len(chunk_words) == 200
        else:
            assert len(chunk_words) == 50


@pytest.mark.asyncio
async def test_store_chromadb_short_text_single_chunk():
    """Short result (≤chunk_size words) produces a single entry with chunk_id='0'."""
    import chromadb
    from src.financial_qa_agent.tools.knowledge_base import _store_in_chromadb

    client = chromadb.Client()
    collection = client.create_collection(
        "test_store_short", metadata={"hnsw:space": "cosine"}
    )

    results = [{"text": "A short article about ETFs.", "source": "https://x.com", "title": "ETF Intro"}]

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._summarize_for_embedding",
        new_callable=AsyncMock,
        return_value="Summary: short ETF article.",
    ):
        await _store_in_chromadb(collection, results)

    stored = collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 1
    assert stored["metadatas"][0]["chunk_id"] == "0"
    assert stored["metadatas"][0]["full_text"] == "A short article about ETFs."


# ---------------------------------------------------------------------------
# Chunk-aware retrieval and formatting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_knowledge_shows_chunk_ids():
    """Retrieval output includes (chunk N) labels for chunked entries."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_chunk_display", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["doc1_chunk_0", "doc1_chunk_1"],
        documents=["Summary of part 1", "Summary of part 2"],
        metadatas=[
            {"source": "https://x.com/article", "title": "Article", "full_text": "Full text of part 1.", "chunk_id": "0"},
            {"source": "https://x.com/article", "title": "Article", "full_text": "Full text of part 2.", "chunk_id": "1"},
        ],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ), patch(
        "src.financial_qa_agent.tools.knowledge_base.settings"
    ) as mock_s:
        mock_s.kb_max_results = 5
        mock_s.kb_max_distance = 0.99
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("article topic")
        assert "(chunk 0)" in result
        assert "(chunk 1)" in result


@pytest.mark.asyncio
async def test_local_knowledge_deduplicates_references():
    """Multiple chunks from same source URL produce Reference line only once."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_dedup_refs", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["a_chunk_0", "a_chunk_1"],
        documents=["Summary chunk 0", "Summary chunk 1"],
        metadatas=[
            {"source": "https://x.com/page", "title": "Page", "full_text": "Chunk zero text.", "chunk_id": "0"},
            {"source": "https://x.com/page", "title": "Page", "full_text": "Chunk one text.", "chunk_id": "1"},
        ],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ), patch(
        "src.financial_qa_agent.tools.knowledge_base.settings"
    ) as mock_s:
        mock_s.kb_max_results = 5
        mock_s.kb_max_distance = 0.99
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("page topic")
        # Reference line should appear only ONCE
        assert result.count("Reference:") == 1
        assert "[Page](https://x.com/page)" in result


@pytest.mark.asyncio
async def test_local_knowledge_backward_compat_no_chunk_id():
    """Old entries without chunk_id metadata render without (chunk N) label."""
    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("test_no_chunk_id", metadata={"hnsw:space": "cosine"})
    # Pre-v0.0.32 entry: has full_text but no chunk_id
    collection.add(
        ids=["old_entry"],
        documents=["Summary of old entry about financial markets"],
        metadatas=[{
            "source": "https://old.com",
            "title": "Old Doc",
            "full_text": "Full text of old document about markets.",
        }],
    )

    with patch(
        "src.financial_qa_agent.tools.knowledge_base._get_collection",
        return_value=collection,
    ), patch(
        "src.financial_qa_agent.tools.knowledge_base.settings"
    ) as mock_s:
        mock_s.kb_max_results = 3
        mock_s.kb_max_distance = 0.99  # Wide threshold to ensure match
        from src.financial_qa_agent.tools.knowledge_base import fetch_local_knowledge

        result = await fetch_local_knowledge("financial markets")
        # Should NOT have chunk label
        assert "(chunk" not in result
        # But should still have the text
        assert "old document about markets" in result.lower()


@pytest.mark.asyncio
async def test_summarize_uses_config_limit():
    """_summarize_for_embedding uses kb_summarize_limit setting, not hardcoded 200."""
    from src.financial_qa_agent.tools.knowledge_base import _summarize_for_embedding

    # Text with 150 words — should be passthrough with default limit=200
    text_150 = " ".join(["word"] * 150)
    result = await _summarize_for_embedding(text_150)
    assert result == text_150  # No LLM call needed

    # Now set a low limit so 150 words triggers LLM
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Short summary."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

    with patch(
        "src.financial_qa_agent.tools.knowledge_base.settings"
    ) as mock_s, patch(
        "src.financial_qa_agent.agent._build_llm",
        return_value=mock_llm,
    ):
        mock_s.kb_summarize_limit = 100  # Lower than 150 words → triggers LLM
        result = await _summarize_for_embedding(text_150)
        assert result == "Short summary."
        mock_llm.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# Fundamental Data Tool — fetch_fundamental_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fundamental_data_no_api_key():
    """Missing FMP API key returns graceful message."""
    with patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s:
        mock_s.fmp_api_key = ""
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data("Apple revenue")
        assert "not configured" in result


@pytest.mark.asyncio
async def test_fundamental_data_no_tickers():
    """No tickers in parse result returns appropriate message."""
    with patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s:
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data("What is revenue?", parse_result={})
        assert "No stock tickers" in result


@pytest.mark.asyncio
async def test_fundamental_data_non_equity_rejected():
    """Non-equity asset types are gracefully rejected."""
    with patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s:
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data(
            "Bitcoin financials",
            parse_result={"tickers": ["BTC-USD"], "asset_type": "crypto"},
        )
        assert "only available for equities" in result


@pytest.mark.asyncio
async def test_fundamental_data_commodity_ticker_filtered():
    """Commodity tickers (GC=F) are filtered out even without asset_type."""
    with patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s:
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data(
            "Gold financials",
            parse_result={"tickers": ["GC=F"]},
        )
        assert "only available for equities" in result


@pytest.mark.asyncio
async def test_fundamental_data_income_statement():
    """Successful income statement fetch returns formatted data."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "date": "2025-09-30",
            "period": "FY",
            "revenue": 394328000000,
            "grossProfit": 170782000000,
            "operatingIncome": 119437000000,
            "netIncome": 96995000000,
            "eps": 6.42,
            "ebitda": 130541000000,
        }
    ]
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s,
        patch(
            "src.financial_qa_agent.tools.fundamental_data.httpx.AsyncClient",
            return_value=mock_client,
        ),
    ):
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data(
            "Apple revenue",
            parse_result={
                "tickers": ["AAPL"],
                "asset_type": "equity",
                "fundamental_endpoints": ["financial_statement"],
            },
        )
        assert "Revenue" in result
        assert "$394.33B" in result
        assert "EPS: $6.42" in result


@pytest.mark.asyncio
async def test_fundamental_data_selects_requested_endpoints():
    """financial_statement makes exactly 1 FMP call (income statement)."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"date": "2025-09-30", "period": "FY", "revenue": 100000000}
    ]
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s,
        patch(
            "src.financial_qa_agent.tools.fundamental_data.httpx.AsyncClient",
            return_value=mock_client,
        ),
    ):
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        await fetch_fundamental_data(
            "Apple revenue",
            parse_result={
                "tickers": ["AAPL"],
                "fundamental_endpoints": ["financial_statement"],
            },
        )
        # financial_statement → 1 FMP call (income statement only)
        assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_fundamental_data_no_endpoints_requested():
    """When parser sets no endpoints, tool returns appropriate message."""
    with patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s:
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data(
            "Apple stock",
            parse_result={
                "tickers": ["AAPL"],
                "fundamental_endpoints": [],
            },
        )
        assert "No fundamental data endpoints requested" in result


@pytest.mark.asyncio
async def test_fundamental_data_earnings_transcript():
    """Earnings transcript is fetched and summarized via LLM."""
    # Mock the FMP API response for transcript
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "date": "2025-07-31",
            "symbol": "AAPL",
            "quarter": 3,
            "year": 2025,
            "content": "Tim Cook: We are pleased to report record revenue of $94.8 billion. "
            "Our services business continues to grow. Looking ahead, we expect "
            "continued strength in our product lineup.",
        }
    ]
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    # Mock the LLM for transcript summarization
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "**Financial Highlights**: Record revenue of $94.8B. "
        "Services business growing. "
        "**Guidance**: Continued strength expected."
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

    with (
        patch("src.financial_qa_agent.tools.fundamental_data.settings") as mock_s,
        patch(
            "src.financial_qa_agent.tools.fundamental_data.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch(
            "src.financial_qa_agent.agent._build_llm",
            return_value=mock_llm,
        ),
    ):
        mock_s.fmp_api_key = "test-key"
        from src.financial_qa_agent.tools.fundamental_data import fetch_fundamental_data

        result = await fetch_fundamental_data(
            "What did Tim Cook say in the latest earnings call?",
            parse_result={
                "tickers": ["AAPL"],
                "fundamental_endpoints": ["earnings_transcript"],
            },
        )
        assert "Earnings Call Transcript" in result
        assert "94.8B" in result
