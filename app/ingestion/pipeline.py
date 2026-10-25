from pathlib import Path

from .chunker import chunk_segments
from .image_loader import image_to_segments
from .models import make_doc_id
from .pdf_loader import load_pdf

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def ingest_bytes(data: bytes, filename: str, chunk_size: int = 800, overlap: int = 120, describer=None):
    """File bytes in -> (doc_id, list[Chunk]) out."""
    ext = Path(filename).suffix.lower()          # "Report.PDF" -> ".pdf"
    doc_id = make_doc_id(data)                   # same file -> same id

    if ext == ".pdf":
        segments = load_pdf(data, filename, doc_id, describer)
    elif ext in IMAGE_EXT:
        segments = image_to_segments(data, filename, doc_id, describer=describer)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return doc_id, chunk_segments(segments, chunk_size, overlap)