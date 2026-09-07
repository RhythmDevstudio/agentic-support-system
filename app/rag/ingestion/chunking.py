"""Structure-aware chunking.

Two rules shape this implementation:

1. **Never merge across provenance boundaries.** A chunk that spanned two PDF
   pages could not honestly report a page number, so page and heading boundaries
   terminate a chunk even when there is budget left.
2. **Split on semantic seams first.** Paragraphs, then sentences, then - only as
   a last resort for pathological input - a hard character cut. Splitting
   mid-sentence damages both retrieval and the quality of a quoted citation.
"""

from __future__ import annotations

import re

from app.models.documents import Chunk, Document, ParsedDocument, ParsedPage, content_hash

# Average characters per token for English prose. Deliberately a heuristic:
# it avoids a tokenizer dependency, and chunk sizing does not need exactness.
_CHARS_PER_TOKEN = 4

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# Sentence boundary: punctuation + whitespace + capital/digit. Avoids splitting
# on "e.g." and on decimal numbers.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9ऀ-ॿ])")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Consistently applied, so relative sizing holds."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort splitter for a single unbroken run longer than the budget."""
    return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)]


def _pack(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack units into chunks, carrying trailing overlap between them."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)

        if unit_tokens > max_tokens:
            # Flush what we have, then split the oversized unit on its own.
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            for piece in _hard_split(unit, max_tokens * _CHARS_PER_TOKEN):
                if piece:
                    chunks.append(piece)
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            carry = _tail_for_overlap(current, overlap_tokens)
            current = [*carry, unit]
            current_tokens = sum(estimate_tokens(part) for part in current)
        else:
            current.append(unit)
            current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def _tail_for_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    """Take trailing context up to - and never beyond - the overlap budget.

    Whole units are preferred so the repeated text stays readable. When even the
    final unit is larger than the budget, a truncated tail of it is carried
    instead: without that fallback, documents made of large paragraphs would get
    no overlap at all, and seam-spanning facts would become unretrievable.

    The budget is a hard ceiling. Carrying an oversized unit would push the next
    chunk past `max_tokens`.
    """
    if overlap_tokens <= 0 or not units:
        return []

    carried: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if total + unit_tokens > overlap_tokens:
            break
        carried.insert(0, unit)
        total += unit_tokens

    if carried:
        return carried

    tail = units[-1][-(overlap_tokens * _CHARS_PER_TOKEN) :]
    # Snap forward to a word boundary so the carried text does not begin mid-word.
    boundary = tail.find(" ")
    if boundary != -1:
        tail = tail[boundary + 1 :]
    return [tail.strip()] if tail.strip() else []


def chunk_page(page: ParsedPage, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Chunk one page/block. Never returns text spanning outside this page."""
    text = page.text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) <= 1:
        paragraphs = _split_sentences(text) or [text]

    # A single paragraph may still exceed the budget; break it into sentences.
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) > max_tokens:
            units.extend(_split_sentences(paragraph) or [paragraph])
        else:
            units.append(paragraph)

    return _pack(units, max_tokens, overlap_tokens)


def chunk_document(
    parsed: ParsedDocument,
    document: Document,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 120,
) -> list[Chunk]:
    """Turn a parsed document into retrievable chunks with full provenance."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[Chunk] = []
    ordinal = 0

    for page in parsed.pages:
        for text in chunk_page(page, max_tokens, overlap_tokens):
            section = page.heading_path[-1] if page.heading_path else None
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document.document_id, ordinal, text),
                    document_id=document.document_id,
                    content=text,
                    ordinal=ordinal,
                    document_title=document.title,
                    source_type=document.source_type,
                    source_path=document.source_path,
                    source_url=document.source_url,
                    publisher=document.publisher,
                    page=page.page,
                    section=section,
                    heading_path=page.heading_path,
                    version=document.version,
                    token_estimate=estimate_tokens(text),
                    checksum=content_hash(text),
                    metadata={**page.metadata},
                )
            )
            ordinal += 1

    return chunks
