import argparse
from urllib.parse import urldefrag, urlsplit

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from ingestion import split_documents


MAX_PAGE_BYTES = 5 * 1024 * 1024

USER_AGENT = (
    "Task6KnowledgeLoader/1.0 "
    "(https://github.com/Abubaker-Nafe)"
)


def html_to_document(html: bytes, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else url
    )

    hostname = (urlsplit(url).hostname or "").lower()

    is_wikipedia = (
        hostname == "wikipedia.org"
        or hostname.endswith(".wikipedia.org")
    )

    # Prefer Wikipedia's article body.
    root = (
        soup.select_one(".mw-parser-output")
        if is_wikipedia
        else None
    )

    # Otherwise prefer the page's main content.
    if root is None:
        root = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

    # Remove common navigation and non-content elements.
    for tag in root.select(
        "script, style, noscript, nav, "
        "footer, header, aside, form"
    ):
        tag.decompose()

    if is_wikipedia:
        for tag in root.select(
            ".mw-editsection, .navbox, "
            ".reflist, sup.reference"
        ):
            tag.decompose()

    text = root.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError(
            "No readable text was found on this page."
        )

    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": title,
            "source_type": (
                "wikipedia" if is_wikipedia else "web"
            ),
        },
    )


def load_web_page(url: str) -> list[Document]:
    url = urldefrag(url.strip())[0]
    parsed = urlsplit(url)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
    ):
        raise ValueError(
            "Provide a complete http:// or https:// URL."
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Use a public URL without embedded credentials."
        )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html, application/xhtml+xml",
    }

    try:
        with requests.get(
            url,
            headers=headers,
            timeout=(5, 20),
            stream=True,
        ) as response:
            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .strip()
                .lower()
            )

            if content_type not in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise ValueError(
                    "Expected an HTML web page, but received: "
                    f"{content_type or 'unknown content type'}"
                )

            parts = []
            total_bytes = 0

            for part in response.iter_content(
                chunk_size=65536
            ):
                if not part:
                    continue

                total_bytes += len(part)

                if total_bytes > MAX_PAGE_BYTES:
                    raise ValueError(
                        "The page exceeds the 5 MB limit."
                    )

                parts.append(part)

            html = b"".join(parts)
            final_url = response.url

    except requests.RequestException as error:
        raise ValueError(
            f"Could not download {url}: {error}"
        ) from error

    document = html_to_document(html, final_url)

    return [document]


web_ingestion_chain = (
    RunnableLambda(load_web_page)
    | RunnableLambda(split_documents)
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and chunk a web page or Wikipedia article."
    )

    parser.add_argument(
        "url",
        help="The public HTML page to load.",
    )

    args = parser.parse_args()

    chunks = web_ingestion_chain.invoke(args.url)

    print("Chunks created:", len(chunks))

    for number, chunk in enumerate(chunks[:2], start=1):
        print(f"\nChunk {number}")
        print("Metadata:", chunk.metadata)
        print("Text:", chunk.page_content[:400])