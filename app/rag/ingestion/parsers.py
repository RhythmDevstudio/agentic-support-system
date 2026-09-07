"""Document parsers: PDF, Markdown, HTML and plain text.

Every parser emits `ParsedPage` blocks that carry the provenance the format can
supply - page numbers for PDF, heading hierarchy for Markdown and HTML. That
metadata is what later becomes the "p. 4, Refund Windows" part of a citation, so
it is captured at parse time and never reconstructed by guesswork.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.models.documents import DocumentFormat, ParsedDocument, ParsedPage
from app.observability.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
    ".txt": DocumentFormat.TEXT,
    ".text": DocumentFormat.TEXT,
    ".faq": DocumentFormat.TEXT,
}


class ParserError(RuntimeError):
    """Raised when a document cannot be parsed."""


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Escapes rather than literals: these characters are invisible in a source file,
# and NO-BREAK SPACE is indistinguishable from a normal space when reading.
_WHITESPACE_RUNS = re.compile("[ \\t\\xa0]+")  # space, tab, NO-BREAK SPACE
_BLANK_LINE_RUNS = re.compile(r"\n{3,}")
# Soft hyphen, zero-width space/non-joiner/joiner, and BOM.
_INVISIBLE = re.compile("[\\xad\\u200b\\u200c\\u200d\\ufeff]")
# Hyphenation introduced by PDF line wrapping: "config-\nuration" -> "configuration".
_PDF_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip invisible characters without losing structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _PDF_HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINE_RUNS.sub("\n\n", text)
    return text.strip()


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def parse_text(content: str, *, title: str, source_path: str | None = None) -> ParsedDocument:
    cleaned = clean_text(content)
    return ParsedDocument(
        title=title,
        document_format=DocumentFormat.TEXT,
        pages=(ParsedPage(text=cleaned),) if cleaned else (),
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(
    content: str, *, title: str | None = None, source_path: str | None = None
) -> ParsedDocument:
    """Split Markdown into blocks at heading boundaries, tracking heading depth.

    Fenced code blocks are tracked so a `#` comment inside a shell snippet is not
    mistaken for a heading.
    """
    lines = content.replace("\r\n", "\n").split("\n")

    pages: list[ParsedPage] = []
    heading_stack: list[str] = []
    buffer: list[str] = []
    current_path: tuple[str, ...] = ()
    doc_title = title
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal buffer
        body = clean_text("\n".join(buffer))
        if body:
            pages.append(ParsedPage(text=body, heading_path=current_path))
        buffer = []

    for line in lines:
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            buffer.append(line)
            continue

        heading = None if in_fence else _ATX_HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and doc_title is None:
                doc_title = text
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(text)
            current_path = tuple(part for part in heading_stack if part)
            # Keep the heading in the body so retrieval can match on it.
            buffer.append(text)
        else:
            buffer.append(line)

    flush()

    return ParsedDocument(
        title=doc_title or (_title_from_path(Path(source_path)) if source_path else "Untitled"),
        document_format=DocumentFormat.MARKDOWN,
        pages=tuple(pages),
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_NON_CONTENT_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "form")


def parse_html(
    content: str,
    *,
    title: str | None = None,
    source_path: str | None = None,
    source_url: str | None = None,
) -> ParsedDocument:
    """Extract readable content from HTML, discarding chrome and scripts."""
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError as exc:  # pragma: no cover - bs4 is a hard dependency
        raise ParserError("beautifulsoup4 is required to parse HTML") from exc

    soup = BeautifulSoup(content, "html.parser")

    for tag in soup.find_all(_NON_CONTENT_TAGS):
        tag.decompose()
    # Comments can carry stale or contradictory text that would pollute retrieval.
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    doc_title = title
    if doc_title is None and soup.title and soup.title.string:
        doc_title = soup.title.string.strip()

    # Prefer the main content region when the page declares one.
    root = soup.find("main") or soup.find("article") or soup.body or soup

    pages: list[ParsedPage] = []
    heading_stack: list[str] = []
    buffer: list[str] = []
    current_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal buffer
        body = clean_text("\n".join(buffer))
        if body:
            pages.append(ParsedPage(text=body, heading_path=current_path))
        buffer = []

    elements = root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "td", "th", "dd", "dt"]
    )
    if not elements:
        text = clean_text(root.get_text("\n"))
        return ParsedDocument(
            title=doc_title or "Untitled",
            document_format=DocumentFormat.HTML,
            pages=(ParsedPage(text=text),) if text else (),
            source_path=source_path,
            source_url=source_url,
        )

    for element in elements:
        name = element.name or ""
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            flush()
            level = int(name[1])
            if level == 1 and doc_title is None:
                doc_title = text
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(text)
            current_path = tuple(part for part in heading_stack if part)
            buffer.append(text)
        else:
            buffer.append(text)

    flush()

    return ParsedDocument(
        title=doc_title or "Untitled",
        document_format=DocumentFormat.HTML,
        pages=tuple(pages),
        source_path=source_path,
        source_url=source_url,
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def parse_pdf(path: Path, *, title: str | None = None) -> ParsedDocument:
    """Parse a PDF, keeping one block per page so page numbers stay citable."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise ParserError("pypdf is required to parse PDF documents") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParserError(f"Could not read PDF {path.name}: {exc}") from exc

    pages: list[ParsedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            # One unreadable page must not fail the whole document.
            logger.warning("pdf_page_extract_failed", path=str(path), page=index, error=str(exc))
            continue
        text = clean_text(raw)
        if text:
            pages.append(ParsedPage(text=text, page=index))

    doc_title = title
    if doc_title is None:
        metadata = getattr(reader, "metadata", None)
        raw_title = getattr(metadata, "title", None) if metadata else None
        doc_title = (raw_title or "").strip() or _title_from_path(path)

    return ParsedDocument(
        title=doc_title,
        document_format=DocumentFormat.PDF,
        pages=tuple(pages),
        source_path=str(path),
        metadata={"page_count": len(reader.pages)},
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def parse_file(path: Path) -> ParsedDocument:
    """Parse any supported file by extension."""
    if not path.exists():
        raise ParserError(f"File not found: {path}")

    document_format = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if document_format is None:
        raise ParserError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if document_format is DocumentFormat.PDF:
        return parse_pdf(path)

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")
        logger.warning("document_decoded_with_replacement", path=str(path))

    if document_format is DocumentFormat.MARKDOWN:
        return parse_markdown(content, source_path=str(path))
    if document_format is DocumentFormat.HTML:
        return parse_html(content, source_path=str(path))
    return parse_text(content, title=_title_from_path(path), source_path=str(path))


def parse_content(
    content: str,
    *,
    document_format: DocumentFormat,
    title: str,
    source_url: str | None = None,
) -> ParsedDocument:
    """Parse in-memory content, used for API uploads and fetched web pages."""
    if document_format is DocumentFormat.MARKDOWN:
        return parse_markdown(content, title=title)
    if document_format is DocumentFormat.HTML:
        return parse_html(content, title=title, source_url=source_url)
    if document_format is DocumentFormat.PDF:
        raise ParserError("PDF content must be parsed from a file path, not a string")
    return parse_text(content, title=title)
