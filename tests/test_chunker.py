from app.ingestion.chunker import chunk_segments, normalize, split_text
from app.ingestion.models import Segment


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ", 100, 10) == []


def test_chunks_respect_size_and_lose_no_words():
    words = [f"word{i}" for i in range(500)]
    pieces = split_text(" ".join(words), chunk_size=200, overlap=40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    joined = " ".join(pieces)
    assert all(w in joined for w in words)


def test_consecutive_chunks_overlap():
    a, b = split_text(" ".join(f"tok{i}" for i in range(300)), 200, 50)[:2]
    assert a.split()[-1] in b.split()


def test_chunks_never_start_mid_word():
    for piece in split_text(" ".join(f"alpha{i}" for i in range(400)), 150, 40):
        assert piece.split()[0].startswith("alpha")


def test_normalize_joins_hyphenated_words():
    assert normalize("retri-\neval") == "retrieval"


def test_chunks_keep_their_page_and_ids_are_stable():
    segs = [Segment("Page one text. " * 80, "a.pdf", "d1", page=1),
            Segment("Page two.", "a.pdf", "d1", page=2)]
    chunks = chunk_segments(segs, 300, 50)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].citation() == "a.pdf (p. 2)"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_segments(segs, 300, 50)]