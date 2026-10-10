"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from raglab import plotting
from raglab.plotting import DATASET_LABEL, INK, INK_2, MUTED, SERIES, SURFACE

# Takeaway titles, written after reading the results.
TITLES = {
    "main": "A dev-tuned hybrid is the best first stage everywhere; only GTE reranking improves on it (FiQA, SciFact)",
    "alpha": "Tuned on dev queries, α = 0.7–0.8 lands on the test optimum for all three corpora",
    "rerank": "Reranking helps only with the right model: GTE gains with depth on FiQA, the others hurt everywhere",
    "chunking": "Small chunks help long abstracts (most when the encoder truncates them) and hurt short FiQA posts",
    "ann": "HNSW keeps 99% of neighbours at 0.4 ms vs 16 ms exact; PQ compression loses half",
    "specificity": "Dense beats BM25 at every level of query specificity (rare-term hypothesis not supported)",
}

METHOD_ORDER = [
    "BM25",
    "MiniLM-L6",
    "BGE-small",
    "BGE-base",
    "E5-base",
    "Hybrid (RRF)",
    "Hybrid (weighted)",
    "Hybrid + MiniLM-L6 cross-encoder",
    "Hybrid + BGE-reranker-base",
    "Hybrid + GTE-reranker-ModernBERT",
]


def _suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def fig_main(summary: pd.DataFrame, names: list[str], path: Path) -> Path:
    methods = [m for m in METHOD_ORDER if m in set(summary.method)]
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.8), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, name in zip(np.atleast_1d(axes), names):
        sub = summary[summary.dataset == name].set_index("method").loc[methods]
        ax.barh(y, sub["ndcg@10"], color=[plotting.FAMILY_COLOR[f] for f in sub.family], height=0.62,
                edgecolor=SURFACE, linewidth=2)
        best = sub["ndcg@10"].max()
        for yi, v in zip(y, sub["ndcg@10"]):
            top = np.isclose(v, best)
            ax.text(v + best * 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5,
                    color=INK if top else INK_2, fontweight="bold" if top else "normal")
        ax.set_title(DATASET_LABEL[name])
        ax.set_xlim(0, best * 1.2)
        ax.set_xlabel("nDCG@10 (test)")
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].set_yticks(y, methods)
    fig.legend(handles=[Patch(color=c, label=f) for f, c in plotting.FAMILY_COLOR.items()],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    _suptitle(fig, TITLES["main"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    plotting.save(fig, path)
    return path


def fig_alpha(sweep: pd.DataFrame, names: list[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.8))
    for ax, name in zip(np.atleast_1d(axes), names):
        dev = sweep[(sweep.dataset == name) & (sweep.split == "dev")]
        test = sweep[(sweep.dataset == name) & (sweep.split == "test")]
        ax.plot(dev.alpha, dev["ndcg@10"], color=SERIES[0], lw=2)
        ax.plot(test.alpha, test["ndcg@10"], color=SERIES[1], lw=2, ls="--")
        chosen = test[test.chosen].iloc[0]
        ax.plot(chosen.alpha, chosen["ndcg@10"], "o", ms=8, color=SERIES[1], mec=SURFACE, mew=2)
        ax.annotate(f"α = {chosen.alpha:.1f}", (chosen.alpha, chosen["ndcg@10"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(f"{DATASET_LABEL[name]} · dense = {dev.dense_model.iloc[0]}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("α (0 = BM25 only, 1 = dense only)")
    np.atleast_1d(axes)[0].set_ylabel("nDCG@10")
    fig.legend(
        handles=[Line2D([], [], color=SERIES[0], lw=2, label="dev queries (used to choose α)"),
                 Line2D([], [], color=SERIES[1], lw=2, ls="--", label="test queries"),
                 Line2D([], [], color=SERIES[1], marker="o", ls="", ms=8, mec=SURFACE, mew=2, label="chosen α on test")],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03),
    )
    _suptitle(fig, TITLES["alpha"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    plotting.save(fig, path)
    return path


def fig_rerank(depth_df: pd.DataFrame, latency_df: pd.DataFrame, names: list[str], path: Path) -> Path:
    rerankers = list(dict.fromkeys(depth_df.reranker))
    depths = sorted(depth_df.depth.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.9))
    for ax, name in zip(np.atleast_1d(axes), names):
        for i, rr in enumerate(rerankers):
            s = depth_df[(depth_df.dataset == name) & (depth_df.reranker == rr)].set_index("depth").loc[depths]
            ax.plot(range(len(depths)), s["ndcg@10"], "-o", color=SERIES[i], mec=SURFACE, mew=1.5)
        ax.set_xticks(range(len(depths)), ["none" if d == 0 else str(d) for d in depths])
        ax.set_xlabel("Candidates reranked (top-k of hybrid)")
        ax.set_title(DATASET_LABEL[name])
    np.atleast_1d(axes)[0].set_ylabel("nDCG@10 (test)")
    hybrid_lat = latency_df[latency_df.method.str.startswith("Hybrid")]
    handles = []
    for i, rr in enumerate(rerankers):
        p50 = hybrid_lat[hybrid_lat.method.str.endswith(rr)].p50_ms.median()
        handles.append(Line2D([], [], color=SERIES[i], marker="o", mec=SURFACE, mew=1.5,
                              label=f"{rr}  (~{p50:,.0f} ms/query at top-100)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(rerankers), bbox_to_anchor=(0.5, -0.03), fontsize=9)
    _suptitle(fig, TITLES["rerank"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    plotting.save(fig, path)
    return path


def fig_chunking(chunk_df: pd.DataFrame, lengths: pd.DataFrame, names: list[str], path: Path) -> Path:
    models = list(dict.fromkeys(chunk_df.model))
    sizes = sorted(s for s in chunk_df.chunk_words.unique() if s > 0) + [0]
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.9))
    for ax, name in zip(np.atleast_1d(axes), names):
        notes = []
        for i, model in enumerate(models):
            s = chunk_df[(chunk_df.dataset == name) & (chunk_df.model == model)].set_index("chunk_words").loc[sizes]
            ax.plot(range(len(sizes)), s["ndcg@10"], "-o", color=SERIES[i], mec=SURFACE, mew=1.5)
            trunc = lengths[(lengths.dataset == name) & (lengths.model == model)].share_truncated.iloc[0]
            notes.append(f"{model} {trunc:.0%}")
        ax.text(0.5, 1.01, "whole docs truncated by the encoder: " + " · ".join(notes), transform=ax.transAxes,
                fontsize=8, color=MUTED, ha="center", va="bottom")
        ax.set_xticks(range(len(sizes)), [str(s) for s in sizes[:-1]] + ["whole doc"])
        ax.set_xlabel("Chunk size (words, 25% overlap, MaxP)")
        ax.set_title(DATASET_LABEL[name], pad=16)
    np.atleast_1d(axes)[0].set_ylabel("nDCG@10 (test)")
    fig.legend(handles=[Line2D([], [], color=SERIES[i], marker="o", mec=SURFACE, mew=1.5, label=m) for i, m in enumerate(models)],
               loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.03))
    _suptitle(fig, TITLES["chunking"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    plotting.save(fig, path)
    return path


def fig_ann(ann_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, family in enumerate(dict.fromkeys(ann_df["index"])):
        s = ann_df[ann_df["index"] == family]
        ax.plot(s.p50_ms, s["neighbour_recall@10"], "-o", color=SERIES[i], mec=SURFACE, mew=1.5,
                label=f"{family}  ({s.size_mb.iloc[0]:.0f} MB)")
    ax.set_xscale("log")
    ax.set_xlabel("Median search latency per query (ms, 1 CPU thread, log scale)")
    ax.set_ylabel("Recall@10 vs exact search")
    n, d = int(ann_df.n_vectors.iloc[0]), int(ann_df.dim.iloc[0])
    ax.set_title(f"FiQA corpus, {n:,} BGE-base vectors ({d}-d); each point is one search setting",
                 loc="left", fontsize=9, fontweight="normal", color=INK_2)
    ax.legend(loc="lower right", title="index (memory)", title_fontsize=8.5)
    _suptitle(fig, TITLES["ann"])
    fig.tight_layout()
    plotting.save(fig, path)
    return path


def fig_specificity(spec_df: pd.DataFrame, names: list[str], dense_labels: dict[str, str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    levels = ["low", "medium", "high"]
    width = 0.26
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        s = spec_df[spec_df.dataset == name]
        means, lo, hi = [], [], []
        for lv in levels:
            diff = (s[s.specificity == lv].dense - s[s.specificity == lv].bm25).to_numpy()
            boots = rng.choice(diff, size=(5000, diff.size)).mean(axis=1)
            means.append(diff.mean())
            lo.append(diff.mean() - np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5) - diff.mean())
        x = np.arange(3) + (i - (len(names) - 1) / 2) * width
        ax.bar(x, means, width=width - 0.03, color=SERIES[i], label=f"{DATASET_LABEL[name]}: {dense_labels[name]} − BM25")
        ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.axhline(0, color=plotting.AXIS, lw=1)
    ax.set_xticks(range(3), ["low\n(only common words)", "medium", "high\n(contains a rare term)"])
    ax.set_xlabel("Query specificity: tercile of the query's rarest-term IDF (95% bootstrap CI)")
    ax.set_ylabel("Δ nDCG@10 (dense − BM25)")
    ax.set_title(TITLES["specificity"], loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figs: Path) -> list[Path]:
    """Redraw every figure from the CSVs in `results`."""
    plotting.apply_style()
    meta = json.loads((results / "run_metadata.json").read_text())
    names = meta["datasets"]
    read = lambda f: pd.read_csv(results / f, dtype={"qid": str})  # noqa: E731
    return [
        fig_main(read("main_results.csv"), names, figs / "01_ndcg_by_method.png"),
        fig_alpha(read("hybrid_alpha_sweep.csv"), names, figs / "02_hybrid_alpha_sweep.png"),
        fig_rerank(read("rerank_depth.csv"), read("rerank_latency.csv"), names, figs / "03_rerank_depth.png"),
        fig_chunking(read("chunking.csv"), read("doc_lengths.csv"), names, figs / "04_chunk_size.png"),
        fig_ann(read("ann_benchmark.csv"), figs / "05_ann_index_tradeoff.png"),
        fig_specificity(read("query_specificity.csv"), names, meta["best_dense_by_dev"], figs / "06_query_specificity.png"),
    ]
