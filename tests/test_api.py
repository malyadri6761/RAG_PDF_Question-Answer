import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.core import services
from app.main import app
from app.retrieval.vector_store import VectorStore

from .conftest import FakeEmbedder, FakeLLM

pymupdf = pytest.importorskip("pymupdf")


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 540, 720), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def llm():
    return FakeLLM("Embeddings map text to vectors [1].")


@pytest.fixture
def client(tmp_path, monkeypatch, llm):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(services, "UPLOAD_DIR", uploads)
    store, emb = VectorStore(tmp_path / "index"), FakeEmbedder()
    app.dependency_overrides[services.get_store] = lambda: store
    app.dependency_overrides[services.get_embedder] = lambda: emb
    app.dependency_overrides[services.get_captioner] = lambda: None
    app.dependency_overrides[routes.require_llm] = lambda: llm
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_then_query_then_delete(client, llm):
    pdf = make_pdf(["Embeddings map text to vectors for semantic search. " * 6,
                    "Unrelated page about cooking rice. " * 6])
    r = client.post("/ingest", files={"file": ("notes.pdf", pdf, "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] == [1, 2]
    assert body["preview"][0]["citation"] == "notes.pdf (p. 1)"

    q = client.post("/query", json={"question": "what do embeddings map text to?"})
    assert q.status_code == 200, q.text
    assert q.json()["citations"][0]["citation"] == "notes.pdf (p. 1)"

    client.post("/ingest", files={"file": ("notes.pdf", pdf, "application/pdf")})   # same file again
    docs = client.get("/documents").json()
    assert len(docs) == 1 and docs[0]["num_chunks"] == body["num_chunks"]

    assert client.delete(f"/documents/{body['doc_id']}").status_code == 200
    assert client.get("/documents").json() == []


def test_query_before_any_upload_is_400(client):
    assert client.post("/query", json={"question": "anything"}).status_code == 400


def test_unsupported_file_is_415(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415