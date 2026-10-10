import numpy as np
import pytest

from raglab.chunking import build_chunks, chunk_words, maxp


def test_short_text_is_one_chunk():
    assert chunk_words("one two three", size=5, overlap=1) == ["one two three"]


def test_chunks_cover_text_with_overlap():
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_words(" ".join(words), size=4, overlap=1)
    assert chunks[0].split() == words[0:4]
    assert chunks[1].split()[0] == words[3]  # one-word overlap
    assert chunks[-1].split()[-1] == words[-1]  # nothing dropped at the end
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_words("a b c", size=3, overlap=3)


def test_build_chunks_prefixes_title_and_tracks_owner():
    owner, texts = build_chunks(["d0", "d1"], ["T0", "T1"], ["a b c d e", "x"], size=3, overlap=0)
    assert owner == [0, 0, 1]
    assert texts[0].startswith("T0 ") and texts[-1] == "T1 x"


def test_maxp_takes_best_chunk_per_document():
    owner = np.array([0, 0, 1])
    idx = np.array([[1, 2, 0]])
    scores = np.array([[0.9, 0.8, 0.3]])
    run = maxp(scores, idx, owner, ["q"], ["d0", "d1"], k=10)
    assert run["q"] == [("d0", pytest.approx(0.9)), ("d1", pytest.approx(0.8))]
