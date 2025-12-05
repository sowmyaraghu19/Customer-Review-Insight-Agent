# agents/retriever.py

import chromadb
from sentence_transformers import SentenceTransformer


class RetrieverAgent:
    def __init__(self):
        # Connect to the local ChromaDB persistent store
        self.client = chromadb.PersistentClient(path="vectorstore")
        self.collection = self.client.get_collection("reviews")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _to_str(self, x):
        """Normalize product/aspect/query into a single string."""
        if x is None:
            return None
        if isinstance(x, list):
            # Join list elements into a single string
            return " ".join(str(item) for item in x)
        return str(x)

    def retrieve(self, product=None, aspect=None, raw_query=None, top_k=8):
        # 1️⃣ Normalize all parts to strings
        product_str = self._to_str(product)
        aspect_str = self._to_str(aspect)
        query_str = self._to_str(raw_query)

        # 2️⃣ Build final search query string
        query = " ".join(
            [x for x in [product_str, aspect_str, query_str] if x]
        )

        # 3️⃣ Encode and query ChromaDB
        query_embedding = self.embedder.encode([query])

        res = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        # 4️⃣ Make sure keys exist and shape is consistent
        documents = res.get("documents", [[]])
        metadatas = res.get("metadatas", [[]])
        distances = res.get("distances", [[]])

        # 5️⃣ Return in the format orchestrator expects
        return {
            "query": query,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
