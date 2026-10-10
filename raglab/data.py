"""Load BEIR retrieval datasets (corpus, queries, graded qrels) from the Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

# Dev split used for every tuning decision; test is only used for reporting.
# SciFact ships no validation split, so its train queries serve as dev.
DATASETS: dict[str, dict[str, str]] = {
    "scifact": {"dev_split": "train", "domain": "science / claim verification"},
    "nfcorpus": {"dev_split": "validation", "domain": "medical / nutrition"},
    "fiqa": {"dev_split": "validation", "domain": "finance Q&A"},
}

Qrels = dict[str, dict[str, int]]  # query id -> doc id -> relevance grade


@dataclass
class BeirDataset:
    """An in-memory BEIR dataset with string ids."""

    name: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, Qrels]  # split name -> qrels

    @property
    def dev_split(self) -> str:
        return DATASETS[self.name]["dev_split"]

    def doc_strings(self) -> list[str]:
        """Title and body joined the way BEIR / MTEB feed documents to encoders."""
        return [f"{t} {x}".strip() for t, x in zip(self.titles, self.texts)]

    def split_queries(self, split: str) -> dict[str, str]:
        """Queries of a split that have at least one relevant document."""
        qrels = self.qrels[split]
        return {
            qid: self.queries[qid]
            for qid in sorted(qrels)
            if qid in self.queries and any(g > 0 for g in qrels[qid].values())
        }


def load_beir(name: str) -> BeirDataset:
    """Download (cached by `datasets`) and normalise one BEIR dataset."""
    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")

    qrels: dict[str, Qrels] = {}
    for split, rows in qrels_ds.items():
        table: Qrels = {}
        for qid, did, score in zip(rows["query-id"], rows["corpus-id"], rows["score"]):
            table.setdefault(str(qid), {})[str(did)] = int(score)
        qrels[split] = table

    return BeirDataset(
        name=name,
        doc_ids=[str(i) for i in corpus["_id"]],
        titles=[t or "" for t in corpus["title"]],
        texts=[t or "" for t in corpus["text"]],
        queries={str(i): t for i, t in zip(queries["_id"], queries["text"])},
        qrels=qrels,
    )
