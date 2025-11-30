import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from preprocess import load_and_preprocess

VECTOR_PATH = "vectorstore"
COLLECTION_NAME = "reviews"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_vectorstore():
    os.makedirs(VECTOR_PATH, exist_ok=True)

    print("\n=== STEP 1: Loading and preprocessing ALL CSV files ===")
    df = load_and_preprocess()

    print("\n=== STEP 2: Embedding review text using MiniLM ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True)

    print("\n=== STEP 3: Creating ChromaDB vectorstore ===")
    client = chromadb.PersistentClient(
        path=VECTOR_PATH,
        settings=Settings(allow_reset=True)
    )
    client.reset()

    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("\n=== STEP 4: Adding documents in safe batches ===")

    batch_size = 5000
    ids = [str(i) for i in df.index]
    docs = df["clean_text"].tolist()
    metas = df[["name", "reviews.rating"]].to_dict(orient="records")
    embeds = embeddings.tolist()

    for i, (id_batch, doc_batch, meta_batch, embed_batch) in enumerate(
        zip(
            chunks(ids, batch_size),
            chunks(docs, batch_size),
            chunks(metas, batch_size),
            chunks(embeds, batch_size),
        )
    ):
        print(f"Adding batch {i + 1}...")
        collection.add(
            ids=id_batch,
            documents=doc_batch,
            metadatas=meta_batch,
            embeddings=embed_batch,
        )

    print("\n✅ Vectorstore successfully built!")
    print("Stored in:", VECTOR_PATH)


if __name__ == "__main__":
    build_vectorstore()
