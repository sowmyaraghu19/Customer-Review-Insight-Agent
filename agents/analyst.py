from statistics import mean
from llm import chat_completion

class AnalystAgent:
    SYSTEM_PROMPT = """
    Analyze summary and ratings:
    - Sentiment label
    - Patterns & complaints
    - Improvement suggestions
    """

    def analyze(self, summary, ratings):
        avg = round(mean(ratings), 2) if ratings else "N/A"
        user_prompt = f"Summary:\n{summary}\nRatings: {ratings}\nAverage: {avg}"
        return chat_completion(self.SYSTEM_PROMPT, user_prompt)
