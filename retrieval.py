from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion import ingestion_chain
from embedding_config import EMBEDDING_MODEL


def build_retriever(chunks: list[Document]):
    if not chunks:
        raise ValueError("No chunks were supplied for indexing.")

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
    )

    print("Embedding and indexing", len(chunks), "chunks...")

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunks)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "data" / "Task-6.pdf"

    chunks = ingestion_chain.invoke(pdf_path)
    retriever = build_retriever(chunks)

    question = (
        "What is the maximum number of iterations "
        "in the feedback loop?"
    )

    results = retriever.invoke(question)

    print("\nQuestion:", question)
    print("Chunks retrieved:", len(results))

    for number, document in enumerate(results, start=1):
        print(f"\nResult {number}")
        print("Metadata:", document.metadata)
        print(document.page_content)
