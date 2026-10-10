"""Approximate nearest-neighbour index benchmarks with FAISS."""

from __future__ import annotations

import time

import faiss
import numpy as np


def build_index(family: str, xb: np.ndarray, **p) -> faiss.Index:
    """Build an inner-product index of the given family on normalised vectors."""
    d = xb.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT
    if family == "Flat":
        index = faiss.IndexFlatIP(d)
    elif family == "HNSW":
        index = faiss.IndexHNSWFlat(d, p.get("M", 32), ip)
        index.hnsw.efConstruction = p.get("efConstruction", 200)
    elif family == "IVF-Flat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, p["nlist"], ip)
    elif family == "IVF-PQ":
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, p["nlist"], p["m"], 8, ip)
    elif family == "IVF-PQ + refine":
        # PQ shortlist of k * k_factor candidates, re-scored with the exact vectors.
        base = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, p["nlist"], p["m"], 8, ip)
        index = faiss.IndexRefineFlat(base)
        index.k_factor = p.get("k_factor", 8)
    elif family == "SQ8":
        index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip)
    else:
        raise ValueError(family)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return index


def set_search_param(index: faiss.Index, name: str | None, value: int) -> None:
    if isinstance(index, faiss.IndexRefine):
        index = faiss.downcast_index(index.base_index)
    if name == "efSearch":
        index.hnsw.efSearch = value
    elif name == "nprobe":
        index.nprobe = value


def index_size_mb(index: faiss.Index) -> float:
    return faiss.serialize_index(index).nbytes / 2**20


def neighbour_recall(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of the exact top-k neighbours recovered by the approximate search."""
    k = truth.shape[1]
    return float(np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)]))


def time_queries(index: faiss.Index, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search one query at a time on a single thread; returns (scores, ids, latency_ms)."""
    lat = np.empty(len(xq))
    all_s, all_i = [], []
    for n, q in enumerate(xq):
        t0 = time.perf_counter()
        s, i = index.search(q[None, :], k)
        lat[n] = (time.perf_counter() - t0) * 1000
        all_s.append(s[0])
        all_i.append(i[0])
    return np.vstack(all_s), np.vstack(all_i), lat
