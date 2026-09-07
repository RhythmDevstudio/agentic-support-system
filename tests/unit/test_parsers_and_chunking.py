"""Phase 2: parsing, cleaning and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.documents import Document, DocumentFormat, ParsedDocument, ParsedPage, SourceType
from app.rag.ingestion.chunking import chunk_document, chunk_page, estimate_tokens
from app.rag.ingestion.parsers import (
    ParserError,
    clean_text,
    parse_file,
    parse_html,
    parse_markdown,
    parse_text,
)


class TestCleanText:
    def test_collapses_whitespace_runs(self) -> None:
        assert clean_text("a    b\t\tc") == "a b c"

    def test_normalises_line_endings(self) -> None:
        assert "\r" not in clean_text("line one\r\nline two\rline three")

    def test_collapses_excess_blank_lines_but_keeps_paragraphs(self) -> None:
        assert clean_text("para one\n\n\n\n\npara two") == "para one\n\npara two"

    def test_removes_invisible_characters(self) -> None:
        assert clean_text("in​visible") == "invisible"

    def test_repairs_pdf_hyphenation(self) -> None:
        """PDF line wrapping splits words; joining them keeps retrieval working."""
        assert "configuration" in clean_text("config-\nuration settings")

    def test_empty_input_is_safe(self) -> None:
        assert clean_text("   \n\n  ") == ""


class TestMarkdownParser:
    def test_builds_heading_hierarchy(self) -> None:
        parsed = parse_markdown(
            "# Title\n\nIntro.\n\n## Section A\n\nBody A.\n\n### Sub A1\n\nBody A1."
        )
        paths = [page.heading_path for page in parsed.pages]
        assert ("Title",) in paths
        assert ("Title", "Section A") in paths
        assert ("Title", "Section A", "Sub A1") in paths

    def test_takes_title_from_first_h1(self) -> None:
        assert parse_markdown("# Refund Policy\n\nBody.").title == "Refund Policy"

    def test_hash_inside_code_fence_is_not_a_heading(self) -> None:
        """A shell comment must not be mistaken for a section boundary."""
        parsed = parse_markdown("# Doc\n\n```bash\n# not a heading\necho hi\n```\n\nAfter.")
        headings = {path[-1] for path in (p.heading_path for p in parsed.pages) if path}
        assert "not a heading" not in headings

    def test_heading_text_is_retained_in_body(self) -> None:
        """Headings carry strong retrieval signal, so they stay in the chunk text."""
        parsed = parse_markdown("# Doc\n\n## Duplicate Charges\n\nBody.")
        assert any("Duplicate Charges" in page.text for page in parsed.pages)

    def test_document_with_no_headings_still_parses(self) -> None:
        parsed = parse_markdown("Just a paragraph.", title="Fallback")
        assert parsed.pages
        assert parsed.title == "Fallback"


class TestHtmlParser:
    def test_strips_scripts_and_navigation(self) -> None:
        parsed = parse_html(
            "<html><head><title>T</title><script>alert(1)</script></head>"
            "<body><nav>Menu</nav><main><p>Real content.</p></main></body></html>"
        )
        text = parsed.full_text
        assert "Real content." in text
        assert "alert" not in text
        assert "Menu" not in text

    def test_extracts_title_tag(self) -> None:
        parsed = parse_html("<html><head><title>Access Guide</title></head><body><p>x</p></body>")
        assert parsed.title == "Access Guide"

    def test_builds_heading_hierarchy(self) -> None:
        parsed = parse_html(
            "<body><h1>Guide</h1><p>a</p><h2>Reset</h2><p>b</p></body>"
        )
        paths = [page.heading_path for page in parsed.pages]
        assert ("Guide",) in paths
        assert ("Guide", "Reset") in paths

    def test_handles_html_without_semantic_tags(self) -> None:
        parsed = parse_html("<body>bare text with no tags</body>")
        assert "bare text" in parsed.full_text


class TestChunking:
    def test_short_page_becomes_one_chunk(self) -> None:
        page = ParsedPage(text="Short body.")
        assert chunk_page(page, max_tokens=800, overlap_tokens=100) == ["Short body."]

    def test_long_page_splits_into_multiple_chunks(self) -> None:
        page = ParsedPage(text="\n\n".join(f"Paragraph number {i}. " * 30 for i in range(20)))
        chunks = chunk_page(page, max_tokens=200, overlap_tokens=40)
        assert len(chunks) > 1
        assert all(estimate_tokens(chunk) <= 260 for chunk in chunks)

    def test_overlap_carries_context_between_chunks(self) -> None:
        units = [f"Sentence block {i} with enough words to matter here." * 6 for i in range(10)]
        chunks = chunk_page(ParsedPage(text="\n\n".join(units)), 120, 40)
        assert len(chunks) > 1
        # The tail of each chunk reappears at the head of the next, so a fact
        # spanning the seam stays retrievable from at least one chunk.
        assert all(chunks[i][-30:] in chunks[i + 1] for i in range(len(chunks) - 1))

    def test_overlap_never_exceeds_its_budget(self) -> None:
        """Carrying an oversized unit would push the next chunk past max_tokens."""
        units = ["A single very long paragraph. " * 40 for _ in range(6)]
        chunks = chunk_page(ParsedPage(text="\n\n".join(units)), 100, 20)
        assert len(chunks) > 1
        assert all(estimate_tokens(chunk) <= 100 + 20 for chunk in chunks)

    def test_oversized_single_paragraph_is_split(self) -> None:
        page = ParsedPage(text="word " * 4000)
        chunks = chunk_page(page, max_tokens=100, overlap_tokens=10)
        assert len(chunks) > 1

    def test_chunks_never_span_pages(self) -> None:
        """A chunk that spanned pages could not honestly report a page number."""
        parsed = ParsedDocument(
            title="Doc",
            document_format=DocumentFormat.PDF,
            pages=(
                ParsedPage(text="Page one content.", page=1),
                ParsedPage(text="Page two content.", page=2),
            ),
        )
        document = Document(document_id="d1", title="Doc")
        chunks = chunk_document(parsed, document, max_tokens=800, overlap_tokens=100)
        assert len(chunks) == 2
        assert {chunk.page for chunk in chunks} == {1, 2}
        assert not any("Page two" in c.content and c.page == 1 for c in chunks)

    def test_provenance_is_preserved_on_every_chunk(self) -> None:
        parsed = ParsedDocument(
            title="Refund Policy",
            document_format=DocumentFormat.MARKDOWN,
            pages=(
                ParsedPage(text="Duplicate charges are refunded in full.",
                           heading_path=("Refund Policy", "Duplicate Charges")),
            ),
        )
        document = Document(
            document_id="d1",
            title="Refund Policy",
            source_type=SourceType.INTERNAL_DOCUMENT,
            source_path="/docs/refund.md",
            publisher="Billing Operations",
            version="3.2",
        )
        chunk = chunk_document(parsed, document)[0]
        assert chunk.document_id == "d1"
        assert chunk.document_title == "Refund Policy"
        assert chunk.section == "Duplicate Charges"
        assert chunk.heading_path == ("Refund Policy", "Duplicate Charges")
        assert chunk.publisher == "Billing Operations"
        assert chunk.version == "3.2"
        assert chunk.source_path == "/docs/refund.md"
        assert chunk.checksum

    def test_chunk_ids_are_deterministic(self) -> None:
        parsed = ParsedDocument(
            title="D", document_format=DocumentFormat.TEXT,
            pages=(ParsedPage(text="stable content"),),
        )
        document = Document(document_id="d1", title="D")
        first = chunk_document(parsed, document)
        second = chunk_document(parsed, document)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_citation_label_combines_page_and_section(self) -> None:
        parsed = ParsedDocument(
            title="D", document_format=DocumentFormat.PDF,
            pages=(ParsedPage(text="body", page=4, heading_path=("D", "Refund Windows")),),
        )
        chunk = chunk_document(parsed, Document(document_id="d1", title="D"))[0]
        assert chunk.citation_label() == "p. 4, Refund Windows"

    def test_rejects_overlap_greater_than_chunk_size(self) -> None:
        parsed = ParsedDocument(
            title="D", document_format=DocumentFormat.TEXT, pages=(ParsedPage(text="x"),)
        )
        with pytest.raises(ValueError, match="overlap_tokens"):
            chunk_document(parsed, Document(document_id="d1", title="D"),
                           max_tokens=100, overlap_tokens=100)


class TestParseFile:
    def test_parses_seed_markdown_document(self, project_root: Path) -> None:
        parsed = parse_file(project_root / "data" / "documents" / "refund_policy.md")
        assert "Refund" in parsed.title
        assert any("duplicate" in page.text.lower() for page in parsed.pages)

    def test_parses_seed_html_document(self, project_root: Path) -> None:
        parsed = parse_file(project_root / "data" / "documents" / "account_access_guide.html")
        assert parsed.document_format is DocumentFormat.HTML
        assert "must not appear" not in parsed.full_text

    def test_parses_seed_text_document(self, project_root: Path) -> None:
        parsed = parse_file(project_root / "data" / "documents" / "billing_faq.txt")
        assert parsed.document_format is DocumentFormat.TEXT
        assert "refund" in parsed.full_text.lower()

    def test_rejects_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xlsx"
        path.write_text("not really a spreadsheet")
        with pytest.raises(ParserError, match="Unsupported file type"):
            parse_file(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ParserError, match="File not found"):
            parse_file(tmp_path / "absent.md")

    def test_text_parser_sets_supplied_title(self) -> None:
        parsed = parse_text("body", title="My Title")
        assert parsed.title == "My Title"
