import logging
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv
from redis import Redis
from redis.exceptions import RedisError

from agent_memory import AgentMemories
from cache_monitor import capture_cache_events
from audio_ingestion import audio_ingestion_chain
from ingestion import TEXT_EXTENSIONS, ingestion_chain
from retrieval import build_retriever
from web_ingestion import web_ingestion_chain


st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task6.ui")

load_dotenv(Path(__file__).with_name(".env"))

st.title("Knowledge Assistant")
st.caption(
    "Ask questions using your documents, code, web pages, "
    "Wikipedia articles, and WAV audio."
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.error(
        "Add OPENROUTER_API_KEY to your .env file, "
        "then restart the app."
    )
    st.stop()

redis_client = Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    socket_connect_timeout=2,
    socket_timeout=2,
)

try:
    redis_client.ping()
except RedisError:
    st.error(
        "Redis is unavailable. Start Docker Desktop "
        "and Redis, then reload."
    )
    st.code("docker compose up -d", language="shell")
    st.stop()
finally:
    redis_client.close()

# Initialize the LLM workflow after the connection checks.
from rag_workflow import build_rag_chain
from workflow import MAX_ITERATIONS


ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".pdf", ".docx", ".ppt", ".pptx", ".wav",
}

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.source_names = []
    st.session_state.chunk_count = 0
    st.session_state.messages = []
    st.session_state.memories = AgentMemories()


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.memories = AgentMemories()


def process_sources(uploads, urls):
    if not uploads and not urls:
        raise ValueError("Upload a file or enter a URL first.")

    if len(uploads) > 10 or len(urls) > 5:
        raise ValueError(
            "Use at most 10 files and 5 URLs at a time."
        )

    if sum(upload.size for upload in uploads) > 50 * 1024 * 1024:
        raise ValueError(
            "The combined file size must not exceed 50 MB."
        )

    chunks = []
    names = []

    with TemporaryDirectory() as directory:
        for number, upload in enumerate(uploads, start=1):
            name = Path(
                upload.name.replace("\\", "/")
            ).name

            extension = Path(name).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file: {name}")

            if not 0 < upload.size <= 20 * 1024 * 1024:
                raise ValueError(
                    f"{name} must be between 1 byte and 20 MB."
                )

            st.write(f"Reading {name}...")

            # Use an app-controlled temporary filename.
            path = (
                Path(directory)
                / f"source_{number}{extension}"
            )

            path.write_bytes(upload.getvalue())

            loader = (
                audio_ingestion_chain
                if extension == ".wav"
                else ingestion_chain
            )

            file_chunks = loader.invoke(path)

            for chunk in file_chunks:
                chunk.metadata["source"] = name

            chunks.extend(file_chunks)
            names.append(name)

    for url in urls:
        st.write(f"Reading {url}...")
        chunks.extend(web_ingestion_chain.invoke(url))
        names.append(url)

    if not chunks:
        raise ValueError("No readable content was found.")

    if len(chunks) > 2000:
        raise ValueError(
            "These sources contain too much text. "
            "Use fewer or smaller sources."
        )

    st.write("Preparing the knowledge base...")
    retriever = build_retriever(chunks)

    return retriever, names, len(chunks)


def show_cache_details(events) -> None:
    with st.expander("Cache"):
        if not events:
            st.caption("Cache information was not recorded for this answer.")
            return
        st.table([
            {
                "Round": event["iteration"],
                "Agent": event["agent"],
                "Status": event["status"],
            }
            for event in events
        ])
        hits = sum(event["status"] == "HIT" for event in events)
        misses = sum(event["status"] == "MISS" for event in events)
        st.caption(f"{hits} hits · {misses} misses")
        st.caption(
            "Cache matches the full request, including conversation history. "
            "Repeating a question can still produce a miss."
        )


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            show_cache_details(message.get("cache_events"))
            return

        result = message.get("result")

        if result and not result["approved"]:
            st.warning(
                "The evaluator did not approve this answer "
                f"after {result['iterations']} rounds."
            )

        st.markdown(message["content"])

        if result:
            decision = (
                "Approved"
                if result["approved"]
                else "Not approved"
            )

            st.caption(
                f"{decision} · "
                f"{result['iterations']}/{MAX_ITERATIONS} rounds"
                f" · {result['elapsed_seconds']:.1f} seconds"
            )

            with st.expander("Evaluation feedback"):
                st.write(result["feedback"])

            show_cache_details(result.get("cache_events"))

            with st.expander("Sources"):
                for source in result["retrieved_sources"]:
                    st.write(
                        f"Passage {source['passage']}: "
                        f"{source.get('source', 'Unknown source')}"
                    )

                    locations = []

                    if "page" in source:
                        locations.append(
                            f"Page {source['page']}"
                        )

                    if "slide" in source:
                        locations.append(
                            f"Slide {source['slide']}"
                        )

                    if "start_seconds" in source:
                        locations.append(
                            f"{source['start_seconds']:.2f}–"
                            f"{source['end_seconds']:.2f} seconds"
                        )

                    if locations:
                        st.caption(" · ".join(locations))


with st.sidebar:
    st.header("Knowledge sources")

    uploads = st.file_uploader(
        "Upload files",
        type=[
            extension.lstrip(".")
            for extension in sorted(ALLOWED_EXTENSIONS)
        ],
        accept_multiple_files=True,
        max_upload_size=20,
    )

    st.caption(
        "PPT and PPTX presentations are supported. "
        "Legacy PPT files are converted automatically."
    )

    url_text = st.text_area(
        "Web or Wikipedia URLs",
        placeholder="One URL per line",
        height=120,
    )

    urls = list(dict.fromkeys(
        line.strip()
        for line in url_text.splitlines()
        if line.strip()
    ))

    st.caption(
        "Click Process sources after changing your selections. "
        "Processing starts a new conversation."
    )

    if st.button("Process sources", type="primary"):
        status = st.status(
            "Processing sources...",
            expanded=True,
        )

        try:
            with status:
                retriever, names, count = process_sources(
                    uploads,
                    urls,
                )

            st.session_state.retriever = retriever
            st.session_state.source_names = names
            st.session_state.chunk_count = count
            clear_chat()

            status.update(
                label="Knowledge base ready",
                state="complete",
                expanded=False,
            )

            logger.info(
                "Knowledge base ready: sources=%d chunks=%d",
                len(names),
                count,
            )

        except Exception as error:
            status.update(
                label="Processing failed",
                state="error",
            )
            st.error(str(error))
            logger.exception("Source processing failed")

    if st.session_state.retriever is not None:
        st.success(
            f"{len(st.session_state.source_names)} sources ready"
        )

        with st.expander("Loaded sources"):
            for name in st.session_state.source_names:
                st.write(name)

    st.button(
        "Clear conversation",
        on_click=clear_chat,
        disabled=st.session_state.retriever is None,
    )


if st.session_state.retriever is None:
    st.info(
        "Add knowledge sources in the sidebar, "
        "then click Process sources."
    )

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input(
    "Ask a question about your loaded sources",
    max_chars=2000,
    disabled=st.session_state.retriever is None,
    submit_mode="disable",
)

if question and question.strip():
    user_message = {
        "role": "user",
        "content": question.strip(),
    }

    st.session_state.messages.append(user_message)
    show_message(user_message)

    attempt_cache_events = []
    try:
        # Commit agent memories only when the workflow completes.
        attempt_memories = deepcopy(
            st.session_state.memories
        )
        started = perf_counter()

        with st.spinner(
            "Generating and evaluating the answer..."
        ):
            chain = build_rag_chain(
                st.session_state.retriever,
                memories=attempt_memories,
            )

            with capture_cache_events() as attempt_cache_events:
                result = chain.invoke(question.strip())

        result["cache_events"] = list(attempt_cache_events)

        result["elapsed_seconds"] = (
            perf_counter() - started
        )

        st.session_state.memories = attempt_memories

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "result": result,
        }

        logger.info(
            "Answer completed: approved=%s "
            "iterations=%d elapsed=%.2f",
            result["approved"],
            result["iterations"],
            result["elapsed_seconds"],
        )

    except Exception as error:
        assistant_message = {
            "role": "assistant",
            "content": (
                f"Could not complete the request: {error}"
            ),
            "error": True,
            "cache_events": list(attempt_cache_events),
        }
        logger.exception("Answer generation failed")

    st.session_state.messages.append(assistant_message)
    show_message(assistant_message)
