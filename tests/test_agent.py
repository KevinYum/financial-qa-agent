"""Integration tests for the LangGraph agent pipeline."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_agent_ask_empty_question():
    """Empty question raises ValueError (preserved from stub)."""
    from src.financial_qa_agent.agent import FinancialQAAgent

    agent = FinancialQAAgent()
    with pytest.raises(ValueError, match="cannot be empty"):
        await agent.ask("")


@pytest.mark.asyncio
async def test_agent_ask_whitespace_only():
    """Whitespace-only question raises ValueError."""
    from src.financial_qa_agent.agent import FinancialQAAgent

    agent = FinancialQAAgent()
    with pytest.raises(ValueError, match="cannot be empty"):
        await agent.ask("   ")


# ---------------------------------------------------------------------------
# Pipeline routing tests (parse result drives tool selection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_pipeline_knowledge_route():
    """Full pipeline: parse with knowledge_queries → fetch_knowledge → synthesize."""
    mock_parse_resp = MagicMock()
    mock_parse_resp.content = json.dumps({
        "tickers": [],
        "company_names": [],
        "time_period": None,
        "time_start": None,
        "time_end": None,
        "asset_type": None,
        "sector": None,
        "needs_news": False,
        "news_query": None,
        "knowledge_queries": ["What is compound interest?"],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = (
        "Compound interest is interest calculated on both the "
        "initial principal and the accumulated interest."
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_knowledge",
            new_callable=AsyncMock,
            return_value="Compound interest is interest on interest.",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask("What is compound interest?")
        assert "compound interest" in answer.lower()


@pytest.mark.asyncio
async def test_agent_pipeline_market_data_route():
    """Full pipeline: parse with tickers → fetch_market_data → synthesize."""
    mock_parse_resp = MagicMock()
    mock_parse_resp.content = json.dumps({
        "tickers": ["AAPL"],
        "company_names": ["Apple"],
        "time_period": "5d",
        "time_start": None,
        "time_end": None,
        "asset_type": "equity",
        "sector": None,
        "needs_news": False,
        "news_query": None,
        "knowledge_queries": [],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = "AAPL is currently trading at $150.00."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_market_data",
            new_callable=AsyncMock,
            return_value="=== Apple Inc. (AAPL) ===\nCurrent Price: $150.00",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask("What is AAPL stock price?")
        assert "150" in answer


@pytest.mark.asyncio
async def test_agent_pipeline_news_route():
    """Full pipeline: parse with needs_news → fetch_news → synthesize."""
    mock_parse_resp = MagicMock()
    mock_parse_resp.content = json.dumps({
        "tickers": [],
        "company_names": ["Apple"],
        "time_period": None,
        "time_start": None,
        "time_end": None,
        "asset_type": None,
        "sector": None,
        "needs_news": True,
        "news_query": "Apple earnings Q4",
        "knowledge_queries": [],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = "Apple reported strong Q4 earnings."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_news",
            new_callable=AsyncMock,
            return_value="[1] Apple Q4 Earnings Beat\n    Strong performance",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask("Latest Apple earnings news")
        assert "earnings" in answer.lower()


@pytest.mark.asyncio
async def test_agent_pipeline_multi_tool_route():
    """Full pipeline: parse with tickers + news + knowledge → all 3 tools → synthesize."""
    mock_parse_resp = MagicMock()
    mock_parse_resp.content = json.dumps({
        "tickers": ["AAPL"],
        "company_names": ["Apple"],
        "time_period": "5d",
        "time_start": None,
        "time_end": None,
        "asset_type": "equity",
        "sector": None,
        "needs_news": True,
        "news_query": "Apple earnings dividends",
        "knowledge_queries": ["How do dividends compound?"],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = (
        "AAPL is at $150 with strong earnings. "
        "Compound interest applies to reinvested dividends."
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_market_data",
            new_callable=AsyncMock,
            return_value="AAPL: $150",
        ),
        patch(
            "src.financial_qa_agent.agent.fetch_news",
            new_callable=AsyncMock,
            return_value="Apple earnings beat expectations",
        ),
        patch(
            "src.financial_qa_agent.agent.fetch_knowledge",
            new_callable=AsyncMock,
            return_value="Dividends can be reinvested via DRIP.",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask(
            "How does AAPL perform and how do dividends compound?"
        )
        assert "150" in answer


# ---------------------------------------------------------------------------
# Parse node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_node_extracts_entities():
    """Parse node extracts structured entities from question."""
    from src.financial_qa_agent.agent import parse_node

    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "tickers": ["TSLA"],
        "company_names": ["Tesla"],
        "time_period": "3mo",
        "time_start": None,
        "time_end": None,
        "asset_type": "equity",
        "sector": None,
        "needs_news": False,
        "news_query": None,
        "knowledge_queries": [],
    })

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm):
        state = {"question": "Tesla stock price last 3 months"}
        result = await parse_node(state)
        assert result["parse_result"]["tickers"] == ["TSLA"]
        assert result["parse_result"]["time_period"] == "3mo"
        assert result["parse_result"]["needs_news"] is False


@pytest.mark.asyncio
async def test_parse_node_json_failure_returns_empty():
    """Parse node returns empty parse_result when LLM returns non-JSON."""
    from src.financial_qa_agent.agent import parse_node

    mock_resp = MagicMock()
    mock_resp.content = "I don't understand that question"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm):
        state = {"question": "random gibberish"}
        result = await parse_node(state)
        assert result["parse_result"] == {}


@pytest.mark.asyncio
async def test_parse_node_strips_markdown_fencing():
    """Parse node strips markdown code fences around JSON."""
    from src.financial_qa_agent.agent import parse_node

    mock_resp = MagicMock()
    mock_resp.content = (
        '```json\n'
        '{"tickers": [], "company_names": [], "time_period": null, '
        '"time_start": null, "time_end": null, "asset_type": null, '
        '"sector": null, "needs_news": true, "news_query": "Tesla news", '
        '"knowledge_queries": []}\n'
        '```'
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm):
        state = {"question": "Latest Tesla news"}
        result = await parse_node(state)
        assert result["parse_result"]["needs_news"] is True
        assert result["parse_result"]["news_query"] == "Tesla news"


# ---------------------------------------------------------------------------
# Routing logic tests
# ---------------------------------------------------------------------------


def test_route_tickers_only():
    """Tickers in parse result routes to fetch_market_data."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {"tickers": ["AAPL"]}}
    result = route_by_parse_result(state)
    assert result == "fetch_market_data"


def test_route_news_only():
    """needs_news=True routes to fetch_news."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {"needs_news": True}}
    result = route_by_parse_result(state)
    assert result == "fetch_news"


def test_route_knowledge_only():
    """knowledge_queries routes to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {"knowledge_queries": ["What is a bond?"]}}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"


def test_route_multiple_tools():
    """Multiple signals route to multiple tools in parallel."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {
        "parse_result": {
            "tickers": ["AAPL"],
            "needs_news": True,
            "knowledge_queries": ["How do dividends work?"],
        }
    }
    result = route_by_parse_result(state)
    assert isinstance(result, list)
    assert "fetch_market_data" in result
    assert "fetch_news" in result
    assert "fetch_knowledge" in result


def test_route_empty_parse_result_fallback():
    """Empty parse result falls back to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {}}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"


def test_route_no_parse_result_key_fallback():
    """Missing parse_result key falls back to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"
