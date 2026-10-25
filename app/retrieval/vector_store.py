import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ingestion.models import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float          # cosine similarity: 1.0 = same meaning, about 0 = unrelated


class VectorStore:
    def __init__(self, index_dir: Path = Path("data/index")):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.dir / "vectors.npy"
        self.meta_path = self.dir / "chunks.jsonl"
        self.vectors: np.ndarray | None = None    # shape (num_chunks, dim)
        self.chunks: list[Chunk] = []             # same order as the rows of self.vectors

        # Load the saved index, so data survives a server restart
        if self.vec_path.exists() and self.meta_path.exists():
            self.vectors = np.load(self.vec_path)
            with self.meta_path.open(encoding="utf-8") as f:
                self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
            if len(self.chunks) != len(self.vectors):
                raise RuntimeError(f"Index is out of sync. Delete {self.dir} and re-ingest.")

    def _save(self) -> None:
        vectors = self.vectors if self.vectors is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(self.vec_path, vectors)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    def _remove(self, doc_ids: set[str]) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id not in doc_ids]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep]       # keep only these rows
        return removed

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add a document's chunks. If the document is already stored, replace it."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        self._remove({c.doc_id for c in chunks})
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = vectors
        else:
            if self.vectors.shape[1] != vectors.shape[1]:
                raise ValueError(f"Embedding size changed. Delete {self.dir} and re-ingest.")
            self.vectors = np.vstack([self.vectors, vectors])   # new rows at the bottom
        self.chunks.extend(chunks)
        self._save()
        return len(chunks)

    def delete(self, doc_id: str) -> int:
        removed = self._remove({doc_id})
        if removed:
            self._save()
        return removed

    def search(self, query_vec: np.ndarray, k: int = 6, doc_ids: list[str] | None = None) -> list[SearchHit]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vec                         # one score per chunk
        if doc_ids:
            mask = np.array([c.doc_id in doc_ids for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)              # hide other documents
        top = np.argsort(-scores)[:k]                             # indices of the k highest scores
        return [SearchHit(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "num_chunks": 0})
            d["num_chunks"] += 1
        return list(docs.values())

    def __len__(self) -> int:
        return len(self.chunks)