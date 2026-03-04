"""Integration tests for the LangGraph agent pipeline."""

import asyncio
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
async def test_agent_pipeline_news_without_tickers_goes_to_knowledge():
    """Full pipeline: needs_news without tickers → knowledge path (not news)."""
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
    mock_synth_resp.content = (
        "Apple earnings information based on available knowledge.\n\n"
        "Based on general financial knowledge."
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
            return_value="Apple recently reported Q4 earnings.",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask("Latest Apple earnings news")
        assert "apple" in answer.lower()


@pytest.mark.asyncio
async def test_agent_pipeline_tickers_with_news():
    """Full pipeline: tickers + needs_news → market_data + news (analysis path)."""
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
        "news_query": "Apple earnings",
        "knowledge_queries": [],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = (
        "AAPL is currently trading at $150.00 with a P/E of 28.5.\n\n"
        "Apple's strong earnings beat expectations, driving positive sentiment."
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
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        answer = await agent.ask("How is AAPL doing with latest earnings?")
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
    """Tickers in parse result routes to analysis → fetch_market_data."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {"tickers": ["AAPL"]}, "question_type": ""}
    result = route_by_parse_result(state)
    assert result == "fetch_market_data"
    assert state["question_type"] == "analysis"


def test_route_tickers_with_news():
    """Tickers + needs_news routes to analysis → market_data + news (not knowledge)."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {
        "parse_result": {
            "tickers": ["AAPL"],
            "needs_news": True,
            "knowledge_queries": ["How do dividends work?"],
        },
        "question_type": "",
    }
    result = route_by_parse_result(state)
    assert isinstance(result, list)
    assert "fetch_market_data" in result
    assert "fetch_news" in result
    assert "fetch_knowledge" not in result
    assert state["question_type"] == "analysis"


def test_route_tickers_without_news():
    """Tickers without needs_news routes to market_data only."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {
        "parse_result": {"tickers": ["TSLA"], "needs_news": False},
        "question_type": "",
    }
    result = route_by_parse_result(state)
    assert result == "fetch_market_data"
    assert state["question_type"] == "analysis"


def test_route_news_without_tickers_goes_to_knowledge():
    """needs_news=True without tickers routes to knowledge (not news)."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {"needs_news": True}, "question_type": ""}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"
    assert state["question_type"] == "knowledge"


def test_route_knowledge_only():
    """knowledge_queries without tickers routes to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {
        "parse_result": {"knowledge_queries": ["What is a bond?"]},
        "question_type": "",
    }
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"
    assert state["question_type"] == "knowledge"


def test_route_empty_parse_result_fallback():
    """Empty parse result falls back to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"parse_result": {}, "question_type": ""}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"
    assert state["question_type"] == "knowledge"


def test_route_no_parse_result_key_fallback():
    """Missing parse_result key falls back to fetch_knowledge."""
    from src.financial_qa_agent.agent import route_by_parse_result

    state = {"question_type": ""}
    result = route_by_parse_result(state)
    assert result == "fetch_knowledge"
    assert state["question_type"] == "knowledge"


# ---------------------------------------------------------------------------
# ask_with_trace tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_with_trace_emits_events():
    """ask_with_trace pushes trace events to the queue and ends with None sentinel."""
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
        "knowledge_queries": ["What is a stock?"],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = "A stock is a share of ownership in a company."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_knowledge",
            new_callable=AsyncMock,
            return_value="A stock represents ownership.",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        queue: asyncio.Queue = asyncio.Queue()
        answer = await agent.ask_with_trace("What is a stock?", queue)

        # Collect all events
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        # Should have trace events (started/completed pairs) + answer + sentinel
        event_types = [e["event_type"] if e else None for e in events]

        assert "trace" in event_types, "Should contain trace events"
        assert "answer" in event_types, "Should contain an answer event"
        assert events[-1] is None, "Last event must be None sentinel"
        assert "stock" in answer.lower()


@pytest.mark.asyncio
async def test_ask_with_trace_empty_question():
    """ask_with_trace emits error and sentinel for empty question."""
    from src.financial_qa_agent.agent import FinancialQAAgent

    agent = FinancialQAAgent()
    queue: asyncio.Queue = asyncio.Queue()

    with pytest.raises(ValueError, match="cannot be empty"):
        await agent.ask_with_trace("", queue)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    # Should have error event + sentinel
    assert any(e and e.get("event_type") == "error" for e in events)
    assert events[-1] is None, "Sentinel must always be sent"


@pytest.mark.asyncio
async def test_ask_without_trace_is_noop():
    """Regular ask() works fine — _emit_trace is a no-op without a queue."""
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
        "knowledge_queries": ["What is a bond?"],
    })

    mock_synth_resp = MagicMock()
    mock_synth_resp.content = "A bond is a fixed-income instrument."

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[mock_parse_resp, mock_synth_resp]
    )

    with (
        patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm),
        patch(
            "src.financial_qa_agent.agent.fetch_knowledge",
            new_callable=AsyncMock,
            return_value="A bond is a debt instrument.",
        ),
    ):
        from src.financial_qa_agent.agent import FinancialQAAgent

        agent = FinancialQAAgent()
        # No trace queue — _emit_trace should be a silent no-op
        answer = await agent.ask("What is a bond?")
        assert "bond" in answer.lower()


# ---------------------------------------------------------------------------
# Synthesize prompt selection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_uses_analysis_prompt():
    """Synthesize node selects analysis prompt when market_data is present."""
    from src.financial_qa_agent.agent import synthesize_node

    mock_resp = MagicMock()
    mock_resp.content = (
        "AAPL is at $150 with P/E of 28.5.\n\n"
        "The stock shows strong momentum."
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm):
        # question_type is intentionally empty — synthesize derives it from market_data
        state = {
            "question": "How is AAPL doing?",
            "question_type": "",
            "market_data": "AAPL: $150",
            "news_data": "",
            "knowledge_data": "",
        }
        result = await synthesize_node(state)

        # Verify the analysis prompt was used (contains "financial analyst" and section headers)
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "financial analyst" in prompt_text
        assert "## Fact" in prompt_text
        assert "## Analysis" in prompt_text
        assert result["answer"] == mock_resp.content


@pytest.mark.asyncio
async def test_synthesize_uses_knowledge_prompt():
    """Synthesize node selects knowledge prompt when market_data is empty."""
    from src.financial_qa_agent.agent import synthesize_node

    mock_resp = MagicMock()
    mock_resp.content = (
        "Compound interest is interest on interest.\n\n"
        "Based on general financial knowledge."
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)

    with patch("src.financial_qa_agent.agent._build_llm", return_value=mock_llm):
        # question_type is intentionally empty — synthesize derives it from market_data
        state = {
            "question": "What is compound interest?",
            "question_type": "",
            "market_data": "",
            "news_data": "",
            "knowledge_data": "Compound interest is interest on interest.",
        }
        result = await synthesize_node(state)

        # Verify the knowledge prompt was used (contains "financial education" and section headers)
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "financial education" in prompt_text
        assert "## Answer" in prompt_text
        assert "## References" in prompt_text
        assert result["answer"] == mock_resp.content
