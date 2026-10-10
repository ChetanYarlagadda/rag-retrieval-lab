"""Published reference scores used to sanity-check this pipeline's numbers.

* BM25: BEIR paper (Thakur et al., NeurIPS 2021, Table 2), Elasticsearch multi-field BM25.
* Dense models: the official MTEB results repository
  (github.com/embeddings-benchmark/results), fetched at run time.
"""

from __future__ import annotations

import json
import os
import urllib.request

BEIR_PAPER_BM25_NDCG10 = {"scifact": 0.665, "nfcorpus": 0.325, "fiqa": 0.236}

MTEB_TASK = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA2018"}
_API = "https://api.github.com/repos/embeddings-benchmark/results/contents/results/{path}"
_RAW = "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/{path}"


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-retrieval-lab"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_mteb_ndcg10(model_id: str, datasets: list[str]) -> dict[str, dict]:
    """nDCG@10 (test split) reported in the MTEB results repo for each dataset.

    Searches every revision folder of the model (and the `external` folder) for the
    task file; the first match wins. Returns {dataset: {"ndcg@10": float, "source": url}}.
    """
    folder = model_id.replace("/", "__")
    revisions = [e["name"] for e in _get_json(_API.format(path=folder)) if e["type"] == "dir"]
    revisions.sort(key=lambda r: r == "external")  # prefer revision folders
    found: dict[str, dict] = {}
    for rev in revisions:
        files = {e["name"] for e in _get_json(_API.format(path=f"{folder}/{rev}"))}
        for ds in datasets:
            fname = f"{MTEB_TASK[ds]}.json"
            if ds in found or fname not in files:
                continue
            url = _RAW.format(path=f"{folder}/{rev}/{fname}")
            data = _get_json(url)
            scores = data["scores"]["test"]
            entry = scores[0] if isinstance(scores, list) else scores
            found[ds] = {"ndcg@10": float(entry["ndcg_at_10"]), "source": url}
        if len(found) == len(datasets):
            break
    return found
