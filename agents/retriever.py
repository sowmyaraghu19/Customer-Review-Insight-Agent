from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class RetrieverAgent:
    def __init__(self, path="vectorstore", collection_name="reviews"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=path, settings=Settings())
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, product=None, aspect=None, raw_query="", top_k=8):
        query = " ".join([x for x in [product, aspect, raw_query] if x])
        embed = self.model.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[embed], n_results=top_k
        )
        return results
