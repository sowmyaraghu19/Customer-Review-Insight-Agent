from typing import Dict, Any, List

from agents import PlannerAgent, RetrieverAgent, SummarizerAgent, AnalystAgent
from memory import ShortTermMemory, LongTermMemory


class ReviewInsightOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.analyst = AnalystAgent()
        self.long_memory = LongTermMemory()

    def run(self, user_query: str, short_memory: ShortTermMemory) -> Dict[str, Any]:
        plan = self.planner.plan(user_query, short_memory.get_all())
        product = plan.get("product")
        aspect = plan.get("aspect")

        short_memory.update(last_product=product, last_aspect=aspect)
        self.long_memory.add_query(user_query)

        retrieval = self.retriever.retrieve(
            product=product, aspect=aspect, raw_query=user_query, top_k=8
        )

        docs = retrieval["documents"][0]
        metadatas = retrieval["metadatas"][0]
        ratings = [float(m.get("reviews.rating", 0)) for m in metadatas]

        summary = self.summarizer.summarize(
            docs, product=product, aspect=aspect
        )

        analysis = self.analyst.analyze(summary, ratings)

        return {
            "plan": plan,
            "docs": docs,
            "metadatas": metadatas,
            "summary": summary,
            "analysis": analysis,
        }
