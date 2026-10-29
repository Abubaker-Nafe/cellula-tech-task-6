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
        raise ValueError(
            "Convert this .ppt file to .pptx using PowerPoint's "
            "Save As option before loading it."
        )

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


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