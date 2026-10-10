"""Run every experiment end to end.

Writes reports/results/*.csv|json and reports/figures/*.png, and logs each
configuration to MLflow (./mlruns). Document embeddings are cached in
data/cache/, so re-runs only recompute what changed.

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bm25s  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402

from raglab import ann  # noqa: E402
from raglab.chunking import build_chunks, maxp  # noqa: E402
from raglab.data import DATASETS, BeirDataset, load_beir  # noqa: E402
from raglab.figures import make_all  # noqa: E402
from raglab.fusion import rrf, weighted_fusion  # noqa: E402
from raglab.metrics import Run, evaluate, summarize  # noqa: E402
from raglab.published import BEIR_PAPER_BM25_NDCG10, fetch_mteb_ndcg10  # noqa: E402
from raglab.rerank import RERANKERS, Reranker, apply_rerank  # noqa: E402
from raglab.retrievers import EMBEDDERS, BM25Retriever, Embedder, flat_index, search_index  # noqa: E402
from raglab.stats import significance_table  # noqa: E402
from raglab.tracking import init_mlflow, log_artifacts, log_result  # noqa: E402

CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "reports" / "results"
FIGS = ROOT / "reports" / "figures"

DEPTH = 100
ALPHAS = [round(a, 1) for a in np.linspace(0, 1, 11)]
CHUNK_MODELS = ["minilm", "bge-small"]
CHUNK_SIZES = [64, 128, 256]
RERANK_DEPTHS = [10, 20, 50, 100]
LATENCY_SAMPLE = 50  # queries timed per rerank depth
BM25_RERANKER = "gte-modernbert"  # also rerank plain BM25 with this one: does the first stage still matter?

FAMILY = {"BM25": "Lexical", **{s.label: "Dense" for s in EMBEDDERS.values()}}


@dataclass
class DatasetState:
    ds: BeirDataset
    docs: list[str]
    doc_index: dict[str, int]
    queries: dict[str, dict[str, str]]  # "dev" | "test" -> qid -> text
    qrels: dict[str, dict]
    bm25: BM25Retriever
    runs: dict[str, dict[str, Run]] = field(default_factory=lambda: {"dev": {}, "test": {}})
    doc_emb: dict[str, np.ndarray] = field(default_factory=dict)
    q_emb: dict[str, dict[str, np.ndarray]] = field(default_factory=lambda: {"dev": {}, "test": {}})


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- experiments


def prepare(names: list[str], max_queries: int | None = None) -> dict[str, DatasetState]:
    states = {}
    for name in names:
        ds = load_beir(name)
        docs = ds.doc_strings()
        queries = {"dev": ds.split_queries(ds.dev_split), "test": ds.split_queries("test")}
        if max_queries:
            queries = {s: dict(list(q.items())[:max_queries]) for s, q in queries.items()}
        states[name] = DatasetState(
            ds=ds,
            docs=docs,
            doc_index={d: i for i, d in enumerate(ds.doc_ids)},
            queries=queries,
            qrels={"dev": ds.qrels[ds.dev_split], "test": ds.qrels["test"]},
            bm25=BM25Retriever(docs, ds.doc_ids),
        )
        for split, q in states[name].queries.items():
            states[name].runs[split]["BM25"] = states[name].bm25.search(q, DEPTH)
        log(f"{name}: {len(docs):,} docs, dev={len(states[name].queries['dev'])} test={len(states[name].queries['test'])} queries")
    return states


def run_dense_and_chunking(states: dict[str, DatasetState]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode with every bi-encoder; also run the chunking experiment for the small models."""
    speed_rows, chunk_rows, length_rows = [], [], []
    for key, spec in EMBEDDERS.items():
        log(f"dense: loading {spec.model_id}")
        emb = Embedder(spec, CACHE)
        tokenizer = emb.model.tokenizer
        for name, st in states.items():
            st.doc_emb[key] = emb.encode_docs(st.docs, cache_key=name)
            index = flat_index(st.doc_emb[key])
            for split, q in st.queries.items():
                t0 = time.perf_counter()
                st.q_emb[split][key] = emb.encode_queries(list(q.values()))
                q_ms = (time.perf_counter() - t0) * 1000 / len(q)
                st.runs[split][spec.label] = search_index(index, st.q_emb[split][key], list(q), st.ds.doc_ids, DEPTH)

            sample = st.docs[:2000]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            emb.encode_docs(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            docs_per_s = len(sample) / (time.perf_counter() - t0)
            n_tokens = np.array([len(t) for t in tokenizer([spec.doc_prefix + d for d in st.docs], add_special_tokens=True)["input_ids"]])
            speed_rows.append(
                {"dataset": name, "model": spec.label, "dim": st.doc_emb[key].shape[1], "docs_per_s": docs_per_s, "query_ms_batched": q_ms}
            )
            length_rows.append(
                {
                    "dataset": name,
                    "model": spec.label,
                    "max_seq_length": spec.max_seq_length,
                    "median_tokens": float(np.median(n_tokens)),
                    "p90_tokens": float(np.percentile(n_tokens, 90)),
                    "share_truncated": float(np.mean(n_tokens > spec.max_seq_length)),
                }
            )

            if key in CHUNK_MODELS:
                test_q = st.queries["test"]
                for size in CHUNK_SIZES:
                    owner, chunks = build_chunks(st.ds.doc_ids, st.ds.titles, st.ds.texts, size, size // 4)
                    c_emb = emb.encode_docs(chunks, cache_key=f"{name}_chunk{size}")
                    c_index = flat_index(c_emb)
                    scores, idx = c_index.search(st.q_emb["test"][key], 1000)
                    run = maxp(scores, idx, np.asarray(owner), list(test_q), st.ds.doc_ids, DEPTH)
                    m = summarize(evaluate(run, st.qrels["test"], list(test_q)))
                    chunk_rows.append({"dataset": name, "model": spec.label, "chunk_words": size, "n_chunks": len(chunks), **m})
                full = summarize(evaluate(st.runs["test"][spec.label], st.qrels["test"], list(test_q)))
                chunk_rows.append({"dataset": name, "model": spec.label, "chunk_words": 0, "n_chunks": len(st.docs), **full})
            log(f"dense: {spec.label} on {name} done ({docs_per_s:,.0f} docs/s)")
        del emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(speed_rows), pd.DataFrame(chunk_rows), pd.DataFrame(length_rows)


def candidate_scores(st: DatasetState, split: str, dense_key: str) -> tuple[dict, dict]:
    """Exact dense and BM25 scores for the union of both systems' top-100 per query."""
    label = EMBEDDERS[dense_key].label
    dense_c, sparse_c = {}, {}
    for i, (qid, text) in enumerate(st.queries[split].items()):
        cand = sorted({d for d, _ in st.runs[split]["BM25"].get(qid, [])} | {d for d, _ in st.runs[split][label][qid]})
        idx = np.array([st.doc_index[d] for d in cand])
        bm = st.bm25.scores(text)[idx]
        dn = st.doc_emb[dense_key][idx] @ st.q_emb[split][dense_key][i]
        dense_c[qid] = dict(zip(cand, dn.tolist()))
        sparse_c[qid] = dict(zip(cand, bm.tolist()))
    return dense_c, sparse_c


def run_hybrid(states: dict[str, DatasetState], best_dense: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, st in states.items():
        key = best_dense[name]
        label = EMBEDDERS[key].label
        for split in ("dev", "test"):
            st.runs[split]["Hybrid (RRF)"] = rrf([st.runs[split]["BM25"], st.runs[split][label]], k=60, depth=DEPTH)
            dense_c, sparse_c = candidate_scores(st, split, key)
            for a in ALPHAS:
                run = weighted_fusion(dense_c, sparse_c, a, DEPTH)
                st.runs[split][f"alpha={a}"] = run
                m = summarize(evaluate(run, st.qrels[split], list(st.queries[split])))
                rows.append({"dataset": name, "split": split, "dense_model": label, "alpha": a, **m})
        sweep = pd.DataFrame([r for r in rows if r["dataset"] == name and r["split"] == "dev"])
        best_alpha = float(sweep.loc[sweep["ndcg@10"].idxmax(), "alpha"])
        for split in ("dev", "test"):
            st.runs[split]["Hybrid (weighted)"] = st.runs[split][f"alpha={best_alpha}"]
        log(f"hybrid: {name} dense={label} best alpha (dev) = {best_alpha}")
    df = pd.DataFrame(rows)
    df["chosen"] = False
    for name in states:
        dev = df[(df.dataset == name) & (df.split == "dev")]
        df.loc[(df.dataset == name) & (df.alpha == dev.loc[dev["ndcg@10"].idxmax(), "alpha"]), "chosen"] = True
    return df


def run_rerank(states: dict[str, DatasetState]) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth_rows, latency_rows = [], []
    for key, spec in RERANKERS.items():
        log(f"rerank: loading {spec.model_id}")
        rr = Reranker(spec)
        rr.score("warm up", ["warm up document"] * 8)
        for name, st in states.items():
            q = st.queries["test"]
            doc_text = dict(zip(st.ds.doc_ids, st.docs))
            first_stages = ["Hybrid (weighted)"] + (["BM25"] if key == BM25_RERANKER else [])
            for first_stage in first_stages:
                base = st.runs["test"][first_stage]
                scores, lat = rr.score_run(q, base, doc_text, DEPTH)
                method = f"{'Hybrid' if first_stage.startswith('Hybrid') else 'BM25'} + {spec.label}"
                st.runs["test"][method] = apply_rerank(base, scores, DEPTH)
                latency_rows.append(
                    {"dataset": name, "method": method, "depth": DEPTH, "p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95))}
                )
                if first_stage == "Hybrid (weighted)":
                    sample = dict(list(q.items())[:LATENCY_SAMPLE])
                    for depth in RERANK_DEPTHS:
                        m = summarize(evaluate(apply_rerank(base, scores, depth), st.qrels["test"], list(q)))
                        _, lat_d = rr.score_run(sample, base, doc_text, depth)
                        depth_rows.append(
                            {"dataset": name, "reranker": spec.label, "depth": depth, "p50_ms": float(np.median(lat_d)), "p95_ms": float(np.percentile(lat_d, 95)), **m}
                        )
                    m0 = summarize(evaluate(base, st.qrels["test"], list(q)))
                    depth_rows.append({"dataset": name, "reranker": spec.label, "depth": 0, "p50_ms": 0.0, "p95_ms": 0.0, **m0})
            log(f"rerank: {spec.label} on {name} done")
        del rr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(depth_rows), pd.DataFrame(latency_rows)


def run_ann(st: DatasetState, dense_key: str) -> pd.DataFrame:
    """Index trade-offs on the largest corpus; exact search is the ground truth.

    Searches return the top 10 (HNSW raises efSearch to at least k, so a larger
    k would hide the efSearch sweep); quality is neighbour recall@10 and nDCG@10.
    """
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    xb = st.doc_emb[dense_key]
    xq = st.q_emb["test"][dense_key]
    qids = list(st.queries["test"])
    exact = ann.build_index("Flat", xb)
    _, truth, _ = ann.time_queries(exact, xq, 10)
    nlist = 1024
    configs = [
        ("Flat", {}, None, [0]),
        ("SQ8", {}, None, [0]),
        ("HNSW", {"M": 32, "efConstruction": 200}, "efSearch", [16, 32, 64, 128, 256]),
        ("IVF-Flat", {"nlist": nlist}, "nprobe", [1, 4, 16, 64]),
        ("IVF-PQ", {"nlist": nlist, "m": 48}, "nprobe", [1, 4, 16, 64]),
        ("IVF-PQ + refine", {"nlist": nlist, "m": 48, "k_factor": 8}, "nprobe", [1, 4, 16, 64]),
    ]
    rows = []
    for family, params, pname, values in configs:
        t0 = time.perf_counter()
        index = ann.build_index(family, xb, **params)
        build_s = time.perf_counter() - t0
        size = ann.index_size_mb(index)
        for v in values:
            ann.set_search_param(index, pname, v)
            scores, ids, lat = ann.time_queries(index, xq, 10)
            run = {q: [(st.ds.doc_ids[j], float(s)) for j, s in zip(r_i, r_s) if j >= 0] for q, r_i, r_s in zip(qids, ids, scores)}
            rows.append(
                {
                    "index": family,
                    "param": pname or "-",
                    "value": v,
                    "neighbour_recall@10": ann.neighbour_recall(ids, truth),
                    "ndcg@10": summarize(evaluate(run, st.qrels["test"], qids))["ndcg@10"],
                    "p50_ms": float(np.median(lat)),
                    "p95_ms": float(np.percentile(lat, 95)),
                    "size_mb": size,
                    "build_s": build_s,
                }
            )
        log(f"ann: {family} done")
    faiss.omp_set_num_threads(threads)
    df = pd.DataFrame(rows)
    df.insert(0, "n_vectors", xb.shape[0])
    df.insert(1, "dim", xb.shape[1])
    return df


def query_specificity(st: DatasetState, dense_label: str) -> pd.DataFrame:
    """Per-test-query BM25 vs dense nDCG@10 together with the query's rarest-term IDF."""
    stemmer = st.bm25.stemmer
    doc_tokens = bm25s.tokenize(st.docs, stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)
    df_count = Counter(t for toks in doc_tokens for t in set(toks))
    n = len(st.docs)
    q = st.queries["test"]
    pq_bm = evaluate(st.runs["test"]["BM25"], st.qrels["test"], list(q)).set_index("qid")
    pq_dn = evaluate(st.runs["test"][dense_label], st.qrels["test"], list(q)).set_index("qid")
    rows = []
    for qid, text in q.items():
        toks = st.bm25.tokenize(text)
        idfs = [np.log((n + 1) / (df_count.get(t, 0) + 1)) for t in toks]
        rows.append(
            {
                "qid": qid,
                "max_idf": max(idfs) if idfs else 0.0,
                "n_terms": len(toks),
                "bm25": pq_bm.loc[qid, "ndcg@10"],
                "dense": pq_dn.loc[qid, "ndcg@10"],
            }
        )
    out = pd.DataFrame(rows)
    out["specificity"] = pd.qcut(out["max_idf"].rank(method="first"), 3, labels=["low", "medium", "high"])
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--skip-published", action="store_true", help="don't fetch MTEB reference scores")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="smoke test: cap queries per split and write outputs to a scratch folder")
    args = parser.parse_args()

    global RESULTS, FIGS
    if args.max_queries:
        RESULTS = FIGS = ROOT / "data" / "cache" / "smoke"
    for d in (CACHE, RESULTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    init_mlflow(RESULTS if args.max_queries else ROOT)
    names = args.datasets
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    t = time.perf_counter()
    states = prepare(names, args.max_queries)
    timings["load_and_bm25_s"] = time.perf_counter() - t

    t = time.perf_counter()
    speed_df, chunk_df, length_df = run_dense_and_chunking(states)
    timings["dense_and_chunking_s"] = time.perf_counter() - t

    # Pick each dataset's dense model on the dev split only.
    best_dense = {}
    dense_dev = []
    for name, st in states.items():
        scores = {}
        for key, spec in EMBEDDERS.items():
            scores[key] = summarize(evaluate(st.runs["dev"][spec.label], st.qrels["dev"], list(st.queries["dev"])))["ndcg@10"]
            dense_dev.append({"dataset": name, "model": spec.label, "dev_ndcg@10": scores[key]})
        best_dense[name] = max(scores, key=scores.get)
        log(f"best dense on {name} dev: {EMBEDDERS[best_dense[name]].label}")

    t = time.perf_counter()
    sweep_df = run_hybrid(states, best_dense)
    timings["hybrid_s"] = time.perf_counter() - t

    t = time.perf_counter()
    depth_df, rerank_lat_df = run_rerank(states)
    timings["rerank_s"] = time.perf_counter() - t

    # ---- main results table + per-query metrics (test split)
    summary_rows, per_query = [], []
    for name, st in states.items():
        q = list(st.queries["test"])
        for method, run in st.runs["test"].items():
            if method.startswith("alpha="):
                continue
            family = FAMILY.get(method) or ("Hybrid + rerank" if "+" in method and method.startswith("Hybrid") else
                                            "Lexical + rerank" if method.startswith("BM25 +") else "Hybrid")
            pq = evaluate(run, st.qrels["test"], q)
            m = summarize(pq)
            summary_rows.append({"dataset": name, "method": method, "family": family, **m, "n_queries": len(q)})
            pq.insert(0, "method", method)
            pq.insert(0, "dataset", name)
            per_query.append(pq)
            params = {"dataset": name, "method": method, "family": family, "split": "test"}
            if family in ("Hybrid", "Hybrid + rerank"):
                params["dense_model"] = EMBEDDERS[best_dense[name]].label
            if method == "Hybrid (weighted)":
                params["alpha"] = float(sweep_df[(sweep_df.dataset == name) & sweep_df.chosen].alpha.iloc[0])
            log_result(f"{name}/{method}", params, m)
    summary = pd.DataFrame(summary_rows)
    per_query_df = pd.concat(per_query, ignore_index=True)

    # ---- ANN on the largest corpus with BGE-base
    t = time.perf_counter()
    ann_name = "fiqa" if "fiqa" in states else names[-1]
    ann_df = run_ann(states[ann_name], "bge-base")
    timings["ann_s"] = time.perf_counter() - t
    for _, r in ann_df.iterrows():
        log_result(f"ann/{r['index']}/{r['param']}={r['value']}",
                   {"dataset": ann_name, "index": r["index"], "search_param": r["param"], "search_value": r["value"]},
                   {k: r[k] for k in ("neighbour_recall@10", "ndcg@10", "p50_ms", "size_mb")})

    # ---- query specificity analysis (BM25 vs each dataset's best dense model)
    spec_frames = []
    for name, st in states.items():
        f = query_specificity(st, EMBEDDERS[best_dense[name]].label)
        f.insert(0, "dataset", name)
        spec_frames.append(f)
    spec_df = pd.concat(spec_frames, ignore_index=True)

    # ---- significance (paired, per query, nDCG@10; Holm-corrected within dataset)
    sig_df = significance_table(per_query_df, {n: EMBEDDERS[best_dense[n]].label for n in names},
                                RERANKERS[BM25_RERANKER].label)

    # ---- published reference scores
    pub_rows = []
    for name in names:
        ours = summary[(summary.dataset == name) & (summary.method == "BM25")]["ndcg@10"].iloc[0]
        pub_rows.append({"dataset": name, "model": "BM25", "ours": ours, "published": BEIR_PAPER_BM25_NDCG10[name],
                         "source": "BEIR paper (Thakur et al. 2021), Table 2"})
    if not args.skip_published:
        for key, spec in EMBEDDERS.items():
            try:
                ref = fetch_mteb_ndcg10(spec.model_id, names)
            except Exception as exc:  # network issues shouldn't sink the run
                log(f"published: could not fetch MTEB scores for {spec.model_id}: {exc}")
                continue
            for name in names:
                if name in ref:
                    ours = summary[(summary.dataset == name) & (summary.method == spec.label)]["ndcg@10"].iloc[0]
                    pub_rows.append({"dataset": name, "model": spec.label, "ours": ours, "published": ref[name]["ndcg@10"],
                                     "source": ref[name]["source"]})
    pub_df = pd.DataFrame(pub_rows)
    pub_df["diff"] = pub_df.ours - pub_df.published

    # ---- write tables
    summary.to_csv(RESULTS / "main_results.csv", index=False, float_format="%.4f")
    per_query_df.to_csv(RESULTS / "per_query_test.csv", index=False, float_format="%.4f")
    pd.DataFrame(dense_dev).to_csv(RESULTS / "dense_dev_selection.csv", index=False, float_format="%.4f")
    sweep_df.to_csv(RESULTS / "hybrid_alpha_sweep.csv", index=False, float_format="%.4f")
    depth_df.to_csv(RESULTS / "rerank_depth.csv", index=False, float_format="%.4f")
    rerank_lat_df.to_csv(RESULTS / "rerank_latency.csv", index=False, float_format="%.2f")
    chunk_df.to_csv(RESULTS / "chunking.csv", index=False, float_format="%.4f")
    length_df.to_csv(RESULTS / "doc_lengths.csv", index=False, float_format="%.3f")
    speed_df.to_csv(RESULTS / "encoder_speed.csv", index=False, float_format="%.2f")
    ann_df.to_csv(RESULTS / "ann_benchmark.csv", index=False, float_format="%.4f")
    spec_df.to_csv(RESULTS / "query_specificity.csv", index=False, float_format="%.4f")
    sig_df.to_csv(RESULTS / "significance.csv", index=False, float_format="%.4f")
    pub_df.to_csv(RESULTS / "published_comparison.csv", index=False, float_format="%.4f")

    timings["total_s"] = time.perf_counter() - t_all

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": names,
        "best_dense_by_dev": {n: EMBEDDERS[best_dense[n]].label for n in names},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(),
            "platform": platform.platform(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "bm25s": bm25s.__version__,
        },
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    figs = make_all(RESULTS, FIGS)
    log_artifacts("summary", figs + [RESULTS / "main_results.csv"])
    log(f"done in {timings['total_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
