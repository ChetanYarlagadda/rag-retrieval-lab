from fastapi.testclient import TestClient

from app.main import create_app
from app.searcher import Hit


class FakeSearcher:
    def search(self, query: str, k: int, rerank: bool) -> list[Hit]:
        return [Hit(doc_id=str(i), title=f"t{i}", snippet=query, score=1.0 / (i + 1)) for i in range(k)]


def client() -> TestClient:
    return TestClient(create_app(FakeSearcher))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_search_returns_k_hits():
    with client() as c:
        r = c.post("/search", json={"query": "insulin resistance", "k": 3, "rerank": False})
        assert r.status_code == 200
        body = r.json()
        assert [h["doc_id"] for h in body["hits"]] == ["0", "1", "2"]
        assert body["reranked"] is False


def test_search_validates_input():
    with client() as c:
        assert c.post("/search", json={"query": ""}).status_code == 422
        assert c.post("/search", json={"query": "x", "k": 500}).status_code == 422
