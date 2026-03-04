"""Shared test fixtures for mocking external dependencies."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm_response():
    """Factory fixture for creating mock LLM responses."""

    def _make(content: str) -> MagicMock:
        mock_response = MagicMock()
        mock_response.content = content
        return mock_response

    return _make
