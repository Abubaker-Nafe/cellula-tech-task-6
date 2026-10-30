import sys
from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader

from embedding_config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKENS,
    EMBEDDING_MAX_TOKENS,
    get_embedding_tokenizer,
)
from ppt_ingestion import converted_pptx

TEXT_EXTENSIONS = {
    ".txt", ".md",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".sql", ".html", ".css",
    ".json", ".yaml", ".yml",
    ".sh", ".ps1", ".bat",
}


def load_pdf(path: Path) -> list[Document]:
    documents = []

    with path.open("rb") as stream:
        reader = PdfReader(stream)

        if reader.is_encrypted:
            raise ValueError("Use an unencrypted PDF.")

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                    },
                ))

    return documents


def load_text(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Cannot decode {path.name}. Save it using UTF-8 encoding."
        ) from error

    return [
        Document(
            page_content=text,
            metadata={"source": path.name},
        )
    ]


def load_docx(path: Path) -> list[Document]:
    word_document = WordDocument(path)
    parts = []

    # Preserve the order of paragraphs and tables in the document body.
    for block in word_document.iter_inner_content():
        if isinstance(block, Paragraph):
            parts.append(block.text)

        elif isinstance(block, Table):
            for row in block.rows:
                parts.append(
                    " | ".join(cell.text for cell in row.cells)
                )

    text = "\n".join(parts).strip()

    return [
        Document(
            page_content=text,
            metadata={"source": path.name},
        )
    ]


def extract_shape_text(shape) -> str:
    # A group can contain other shapes, including nested groups.
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        return "\n".join(
            extract_shape_text(child)
            for child in shape.shapes
        )

    if shape.has_text_frame:
        return shape.text_frame.text

    if shape.has_table:
        return "\n".join(
            " | ".join(cell.text for cell in row.cells)
            for row in shape.table.rows
        )

    return ""


def load_pptx(path: Path) -> list[Document]:
    presentation = Presentation(path)
    documents = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):
        text = "\n".join(
            extract_shape_text(shape)
            for shape in slide.shapes
        ).strip()

        if text:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "slide": slide_number,
                },
            ))

    return documents


def load_ppt(path: Path) -> list[Document]:
    with converted_pptx(path) as converted:
        documents = load_pptx(converted)
    for document in documents:
        document.metadata["source"] = path.name
    return documents


def load_file(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"The file is empty: {path.name}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        documents = load_pdf(path)

    elif extension == ".docx":
        documents = load_docx(path)

    elif extension == ".pptx":
        documents = load_pptx(path)

    elif extension in TEXT_EXTENSIONS:
        documents = load_text(path)

    elif extension == ".ppt":
        documents = load_ppt(path)

    else:
        raise ValueError(
            f"Unsupported file extension: {extension or '(none)'}"
        )

    documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            f"No extractable text found in {path.name}. "
            "This loader does not perform OCR on images."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    tokenizer = get_embedding_tokenizer()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=lambda text: len(
            tokenizer.encode(
                text, add_special_tokens=False, truncation=False, verbose=False
            )
        ),
        separators=["\n\n", "\n", " ", ""],
    )

    def fitting_parts(text):
        # Count final input including special tokens; never embed a truncated chunk.
        if len(tokenizer.encode(
            text, add_special_tokens=True, truncation=False, verbose=False
        )) <= EMBEDDING_MAX_TOKENS:
            return [text]
        if len(text) < 2:
            raise ValueError("A text fragment exceeds the embedding token limit.")
        midpoint = len(text) // 2
        return fitting_parts(text[:midpoint]) + fitting_parts(text[midpoint:])

    chunks = []
    for document in documents:
        previous_start = -1
        for text in splitter.split_text(document.page_content):
            for part in fitting_parts(text):
                # Token overlap is not a character offset; locate the exact substring.
                start = document.page_content.find(part, previous_start + 1)
                if start < 0:
                    raise ValueError("Could not locate a chunk in its source text.")
                chunks.append(Document(
                    page_content=part,
                    metadata={**document.metadata, "start_index": start},
                ))
                previous_start = start
    return chunks


ingestion_chain = (
    RunnableLambda(load_file)
    | RunnableLambda(split_documents)
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = (
            Path(__file__).parent / "data" / "Task-6.pdf"
        )

    chunks = ingestion_chain.invoke(file_path)

    print("File:", file_path.name)
    print("Chunks created:", len(chunks))

    for number, chunk in enumerate(chunks[:2], start=1):
        print(f"\nChunk {number}")
        print("Metadata:", chunk.metadata)
        print("Text:", chunk.page_content[:400])
