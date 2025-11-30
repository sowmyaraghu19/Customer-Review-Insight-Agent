import json
from llm import chat_completion

class PlannerAgent:
    SYSTEM_PROMPT = """
    You extract:
    - product
    - aspect
    - intent
    Return ONLY JSON.
    """

    def plan(self, user_query: str, memory: dict):
        user_prompt = f"Query: {user_query}\nMemory: {json.dumps(memory)}"
        result = chat_completion(self.SYSTEM_PROMPT, user_prompt)
        try:
            return json.loads(result)
        except:
            return {"product": None, "aspect": None, "intent": result}
