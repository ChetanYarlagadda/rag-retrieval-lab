"""First-stage retrievers: BM25 (bm25s) and bi-encoder dense retrieval (Sentence Transformers + FAISS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bm25s
import faiss
import numpy as np
import Stemmer
import torch
from sentence_transformers import SentenceTransformer

from raglab.metrics import Run

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbedderSpec:
    """A bi-encoder plus the input conventions it was trained with."""

    key: str
    label: str
    model_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int = 512


EMBEDDERS: dict[str, EmbedderSpec] = {
    s.key: s
    for s in [
        EmbedderSpec("minilm", "MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256),
        EmbedderSpec("bge-small", "BGE-small", "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY_INSTRUCTION),
        EmbedderSpec("bge-base", "BGE-base", "BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_INSTRUCTION),
        EmbedderSpec("e5-base", "E5-base", "intfloat/e5-base-v2", "query: ", "passage: "),
    ]
}


class BM25Retriever:
    """Okapi BM25 (Lucene variant, k1=1.5, b=0.75) with English stemming and stopwords."""

    def __init__(self, docs: list[str], doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids
        self.stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.model = bm25s.BM25()
        self.model.index(tokens, show_progress=False)

    def tokenize(self, query: str) -> list[str]:
        return bm25s.tokenize(
            [query], stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False
        )[0]

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for one query (zeros if no indexable terms)."""
        terms = self.tokenize(query)
        if not terms:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return self.model.get_scores(terms).astype(np.float32)

    def search(self, queries: dict[str, str], k: int = 100) -> Run:
        run: Run = {}
        for qid, text in queries.items():
            s = self.scores(text)
            top = np.argsort(-s, kind="stable")[:k]
            run[qid] = [(self.doc_ids[i], float(s[i])) for i in top if s[i] > 0]
        return run


class Embedder:
    """Encodes queries and documents with an EmbedderSpec; caches document embeddings on disk."""

    def __init__(self, spec: EmbedderSpec, cache_dir: Path | None = None, batch_size: int = 64) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 on GPU: several times faster; nDCG@10 moves by <= 0.002 (checked on SciFact).
        self.precision = "fp16" if device == "cuda" else "fp32"
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = SentenceTransformer(spec.model_id, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = spec.max_seq_length

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_docs(self, docs: list[str], cache_key: str | None = None) -> np.ndarray:
        path = None
        if self.cache_dir is not None and cache_key is not None:
            path = self.cache_dir / f"{cache_key}__{self.spec.key}__{self.precision}.npy"
            if path.exists():
                cached = np.load(path)
                if cached.shape[0] == len(docs):
                    return cached
        emb = self._encode([self.spec.doc_prefix + d for d in docs])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + q for q in queries])


def flat_index(doc_emb: np.ndarray) -> faiss.Index:
    """Exact inner-product index (cosine similarity on normalised vectors)."""
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    return index


def search_index(index: faiss.Index, query_emb: np.ndarray, qids: list[str], doc_ids: list[str], k: int = 100) -> Run:
    """Top-k search for a batch of queries, returned as a Run."""
    scores, idx = index.search(query_emb, k)
    return {
        qid: [(doc_ids[j], float(s)) for j, s in zip(row_i, row_s) if j >= 0]
        for qid, row_i, row_s in zip(qids, idx, scores)
    }
