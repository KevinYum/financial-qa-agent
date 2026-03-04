"""Financial QA Agent — processes user financial questions."""


class FinancialQAAgent:
    """Stub agent that answers financial questions.

    Behavior will be defined in future iterations.
    """

    async def ask(self, question: str) -> str:
        """Process a financial question and return an answer."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Placeholder — agent behavior to be defined later
        return (
            f"Thank you for your question: \"{question}\". "
            "The financial QA agent is not yet configured. "
            "Agent behavior will be defined in a future update."
        )
