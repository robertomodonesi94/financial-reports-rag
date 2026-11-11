"""Turn the text of each PDF page into overlapping word-window chunks that remember their page."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    id: str          # "<doc_id>#p<page>-<position>"
    doc_id: str
    title: str
    page: int        # zero-based PDF page, the same convention as FinanceBench evidence
    text: str
    position: int    # position inside the page

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(?=\w)", "", text)        # undo line-break hyphenation
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunks(doc_id: str, title: str, text: str, size: int = 220, overlap: int = 40, page: int = 0) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = [w for w in clean_text(text).split(" ") if w]
    if not words:
        return []
    step = size - overlap
    chunks: list[Chunk] = []
    for pos, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + size]
        chunks.append(Chunk(id=f"{doc_id}#p{page}-{pos}", doc_id=doc_id, title=title, page=page, text=" ".join(piece), position=pos))
        if start + size >= len(words):
            break
    return chunks


def chunk_pages(doc_id: str, title: str, pages: list[str], size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Chunk a document page by page. Chunks never cross a page boundary, so retrieval can be scored per page."""
    out: list[Chunk] = []
    for page, text in enumerate(pages):
        out.extend(make_chunks(doc_id, title, text, size, overlap, page=page))
    return out
