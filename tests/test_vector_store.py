from app.ingestion.models import Chunk
from app.retrieval.vector_store import VectorStore

from .conftest import FakeEmbedder


def chunk(cid, text, doc="d1", page=1, source="a.pdf"):
    return Chunk(cid, text, source, doc, page, "text", 0)


def add(store, emb, chunks):
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))


def test_most_relevant_chunk_ranks_first(tmp_path):
    store, emb = VectorStore(tmp_path), FakeEmbedder()
    add(store, emb, [chunk("c1", "transformers use attention"), chunk("c2", "the invoice total was 500 rupees")])
    hits = store.search(emb.embed_query("what was the invoice total"), k=2)
    assert hits[0].chunk.chunk_id == "c2" and hits[0].score > hits[1].score


def test_index_survives_restart(tmp_path):
    emb = FakeEmbedder()
    add(VectorStore(tmp_path), emb, [chunk("c1", "alpha beta gamma")])
    reloaded = VectorStore(tmp_path)
    assert len(reloaded) == 1
    assert reloaded.search(emb.embed_query("beta"), k=1)[0].chunk.chunk_id == "c1"


def test_reupload_replaces_instead_of_duplicating(tmp_path):
    store, emb = VectorStore(tmp_path), FakeEmbedder()
    add(store, emb, [chunk("c1", "old text"), chunk("c2", "old text two")])
    add(store, emb, [chunk("c9", "new text")])
    assert len(store) == 1


def test_doc_filter_and_delete(tmp_path):
    store, emb = VectorStore(tmp_path), FakeEmbedder()
    add(store, emb, [chunk("c1", "budget numbers", doc="d1")])
    add(store, emb, [chunk("c2", "budget numbers", doc="d2")])
    hits = store.search(emb.embed_query("budget"), k=5, doc_ids=["d2"])
    assert [h.chunk.doc_id for h in hits] == ["d2"]
    assert store.delete("d1") == 1
    assert [d["doc_id"] for d in store.documents()] == ["d2"]