import argparse
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from ingestion import split_documents


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    print("Loading Whisper base model on CPU...")

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )


def load_audio(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    if path.suffix.lower() != ".wav":
        raise ValueError("This loader expects a WAV file.")

    if path.stat().st_size == 0:
        raise ValueError("The audio file is empty.")

    model = get_whisper_model()

    segments, info = model.transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    documents = []

    # Iterating over segments actually runs the transcription.
    for number, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        if text:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_type": "audio",
                    "language": info.language,
                    "segment": number,
                    "start_seconds": round(
                        float(segment.start), 2
                    ),
                    "end_seconds": round(
                        float(segment.end), 2
                    ),
                },
            ))

    if not documents:
        raise ValueError(
            "No speech was transcribed from this audio file."
        )

    print("Detected language:", info.language)

    return documents


audio_ingestion_chain = (
    RunnableLambda(load_audio)
    | RunnableLambda(split_documents)
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe and chunk a WAV audio file."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the WAV file.",
    )

    args = parser.parse_args()

    chunks = audio_ingestion_chain.invoke(args.file)

    print("File:", args.file.name)
    print("Chunks created:", len(chunks))

    for number, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {number}")
        print("Metadata:", chunk.metadata)
        print("Text:", chunk.page_content)