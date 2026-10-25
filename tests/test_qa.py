from app.ingestion.models import Chunk
from app.rag.qa import NOT_FOUND, answer_question, parse_citations
from app.retrieval.vector_store import VectorStore

from .conftest import FakeEmbedder, FakeLLM, make_text_image


def build_store(tmp_path, chunks):
    store, emb = VectorStore(tmp_path / "index"), FakeEmbedder()
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    return store, emb


def test_parse_citations_handles_all_formats():
    assert parse_citations("A [2]. B [1][2]. C [3, 1]. D [9].", max_n=3) == [2, 1, 3]


def test_answer_maps_citation_to_file_and_page(tmp_path):
    store, emb = build_store(tmp_path, [
        Chunk("c1", "the warranty period is two years", "manual.pdf", "d1", 4, "text", 0),
        Chunk("c2", "the battery lasts ten hours", "manual.pdf", "d1", 7, "text", 0),
    ])
    llm = FakeLLM("The warranty lasts two years [1].")
    result = answer_question("how long is the warranty", emb, store, llm, min_score=0.0, upload_dir=None)
    assert [s.citation for s in result.citations] == ["manual.pdf (p. 4)"]
    assert "[1] manual.pdf, page 4" in llm.calls[0]["prompt"]


def test_irrelevant_question_skips_the_llm(tmp_path):
    store, emb = build_store(tmp_path, [Chunk("c1", "photosynthesis in plants", "bio.pdf", "d1", 1, "text", 0)])
    llm = FakeLLM()
    result = answer_question("invoice total rupees", emb, store, llm, min_score=0.3, upload_dir=None)
    assert result.answer == NOT_FOUND and llm.calls == []


def test_original_image_is_sent_to_llm(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    png = make_text_image(["Revenue"])
    (uploads / "img1.png").write_bytes(png)
    store, emb = build_store(tmp_path, [
        Chunk("c1", "bar chart of revenue by quarter", "chart.png", "img1", None, "image_caption", 0)])
    llm = FakeLLM()
    answer_question("revenue by quarter", emb, store, llm, min_score=0.0, upload_dir=uploads)
    assert llm.calls[0]["images"] == [png]