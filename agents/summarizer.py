from llm import chat_completion

class SummarizerAgent:
    SYSTEM_PROMPT = """
    Summarize reviews with:
    - Sentiment
    - Pros
    - Cons
    """

    def summarize(self, reviews, product=None, aspect=None):
        combined = "\n\n---\n".join(reviews)
        user_prompt = f"Product: {product}\nAspect: {aspect}\n\nReviews:\n{combined}"
        return chat_completion(self.SYSTEM_PROMPT, user_prompt)
