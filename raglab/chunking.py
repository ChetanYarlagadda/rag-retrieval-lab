"""Fixed-size word chunking with overlap, and MaxP aggregation back to documents."""

from __future__ import annotations

import numpy as np

from raglab.metrics import Run


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words that overlap by `overlap` words."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(
    doc_ids: list[str], titles: list[str], texts: list[str], size: int, overlap: int
) -> tuple[list[int], list[str]]:
    """Chunk every document body; each chunk is prefixed with its document title.

    Returns (owner, chunk_texts) where owner[i] is the document index of chunk i.
    """
    owner: list[int] = []
    chunk_texts: list[str] = []
    for i, (title, text) in enumerate(zip(titles, texts)):
        for chunk in chunk_words(text, size, overlap):
            owner.append(i)
            chunk_texts.append(f"{title} {chunk}".strip())
    return owner, chunk_texts


def maxp(scores: np.ndarray, idx: np.ndarray, owner: np.ndarray, qids: list[str], doc_ids: list[str], k: int = 100) -> Run:
    """Document score = best chunk score (MaxP). `idx`/`scores` are FAISS chunk hits per query."""
    run: Run = {}
    for qid, row_i, row_s in zip(qids, idx, scores):
        best: dict[int, float] = {}
        for j, s in zip(row_i, row_s):
            if j < 0:
                continue
            d = int(owner[j])
            if s > best.get(d, -np.inf):
                best[d] = float(s)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
        run[qid] = [(doc_ids[d], s) for d, s in ranked]
    return run
