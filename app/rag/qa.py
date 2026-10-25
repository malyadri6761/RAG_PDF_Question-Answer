import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ANSWER_SYSTEM = """You answer questions using ONLY the numbered sources provided.
Rules:
- Cite every factual claim with its source number in square brackets, e.g. [1] or [2][4].
- If the sources do not contain the answer, say you could not find it in the documents. Do not guess.
- If sources disagree, say so and cite both.
- Sources marked "image" come from pictures (OCR text or a model's description); any attached images are the originals.
- Be concise. Do not mention these rules."""

NOT_FOUND = "I couldn't find anything relevant to that in the uploaded documents."
MAX_ATTACHED_IMAGES = 2
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")      # [1]  [2, 3]


@dataclass
class Source:
    number: int           # the [n] used in the prompt and the answer
    citation: str         # "report.pdf (p. 4)"
    doc_id: str
    page: int | None
    content_type: str
    score: float
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Source]                              # sources the answer actually cites
    sources: list[Source] = field(default_factory=list)  # everything that was retrieved

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(hits) -> str:
    """Numbered source blocks for the prompt."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        loc = f"page {c.page}" if c.page is not None else "standalone image"
        kind = "text" if c.content_type in ("text", "page_ocr") else "image"
        blocks.append(f"[{n}] {c.source}, {loc}, {kind}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def parse_citations(answer: str, max_n: int) -> list[int]:
    """Source numbers the answer cites, in order of first use, ignoring invalid ones."""
    found = []
    for group in CITE_RE.findall(answer):
        for num in group.split(","):
            n = int(num)
            if 1 <= n <= max_n and n not in found:
                found.append(n)
    return found


def images_for(hits, upload_dir: Path | None) -> list[bytes]:
    """Original files of standalone-image hits, so the LLM can look at the real picture."""
    if upload_dir is None:
        return []
    images, seen = [], set()
    for h in hits:
        c = h.chunk
        if c.page is not None or c.doc_id in seen:      # only standalone images, each once
            continue
        seen.add(c.doc_id)
        files = list(Path(upload_dir).glob(f"{c.doc_id}.*"))
        if files:
            images.append(files[0].read_bytes())
        if len(images) >= MAX_ATTACHED_IMAGES:
            break
    return images


def answer_question(question, embedder, store, llm, top_k=6, min_score=0.2,
                    doc_ids=None, upload_dir: Path | None = Path("data/uploads")) -> Answer:
    # 1. Retrieve
    hits = store.search(embedder.embed_query(question), k=top_k, doc_ids=doc_ids)
    hits = [h for h in hits if h.score >= min_score]
    if not hits:                                       # nothing relevant: don't let the LLM guess
        return Answer(NOT_FOUND, [], [])

    # 2. Generate
    prompt = f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"
    text = llm.generate(ANSWER_SYSTEM, prompt, images=images_for(hits, upload_dir) or None)

    # 3. Map [n] back to files and pages
    sources = [
        Source(n, h.chunk.citation(), h.chunk.doc_id, h.chunk.page, h.chunk.content_type,
               round(h.score, 3), h.chunk.text[:300])
        for n, h in enumerate(hits, start=1)
    ]
    cited = [sources[n - 1] for n in parse_citations(text, len(sources))]
    return Answer(text, cited, sources)