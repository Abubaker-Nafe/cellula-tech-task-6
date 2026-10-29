import argparse
import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from agent_memory import AgentMemories
from audio_ingestion import audio_ingestion_chain
from ingestion import ingestion_chain
from retrieval import build_retriever
from web_ingestion import web_ingestion_chain
from workflow import run_workflow


def build_rag_chain(
    retriever,
    memories: AgentMemories | None = None,
):
    def retrieve_context(question: str) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Please provide a question.")

        documents = retriever.invoke(question)

        if not documents:
            raise ValueError("No passages were retrieved.")

        context_blocks = []
        retrieved_sources = []

        for number, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)

            context_blocks.append(
                f"[Passage {number}]\n"
                f"Source metadata: "
                f"{json.dumps(metadata, ensure_ascii=False)}\n"
                f"Text:\n{document.page_content}"
            )

            retrieved_sources.append({
                **metadata,
                "passage": number,
            })

        print(f"\nRetrieved {len(documents)} chunks.")

        return {
            "question": question,
            "context": "\n\n".join(context_blocks),
            "retrieved_sources": retrieved_sources,
        }

    def generate_and_evaluate(inputs: dict) -> dict:
        result = run_workflow(
            question=inputs["question"],
            context=inputs["context"],
            memories=memories,
        )

        return {
            **result,
            "retrieved_sources": inputs["retrieved_sources"],
        }

    return (
        RunnableLambda(retrieve_context)
        | RunnableLambda(generate_and_evaluate)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a question using local documents, "
            "WAV audio, and web pages."
        )
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Paths to documents, source code, or WAV files.",
    )

    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=[],
        help="A web or Wikipedia URL. Can be repeated.",
    )

    parser.add_argument(
        "-q",
        "--question",
        default=(
            "What is the maximum number of iterations "
            "in the feedback loop?"
        ),
        help="The question to answer.",
    )

    args = parser.parse_args()

    if not args.question.strip():
        parser.error("The question cannot be empty.")

    file_paths = args.files

    if not file_paths and not args.urls:
        file_paths = [
            Path(__file__).parent / "data" / "Task-6.pdf"
        ]

    chunks = []

    for file_path in file_paths:
        if file_path.suffix.lower() == ".wav":
            file_chunks = audio_ingestion_chain.invoke(
                file_path
            )
        else:
            file_chunks = ingestion_chain.invoke(file_path)

        chunks.extend(file_chunks)

        print(
            f"Loaded {file_path.name}: "
            f"{len(file_chunks)} chunks"
        )

    for url in args.urls:
        web_chunks = web_ingestion_chain.invoke(url)
        chunks.extend(web_chunks)

        print(
            f"Loaded {url}: "
            f"{len(web_chunks)} chunks"
        )

    retriever = build_retriever(chunks)
    rag_chain = build_rag_chain(retriever)

    print("\nQuestion:", args.question)

    result = rag_chain.invoke(args.question)

    print("\nFinal result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()