import logging

import fitz  # PyMuPDF

from .image_loader import image_to_segments, ocr_image
from . import Segment

log = logging.getLogger(__name__)

SCAN_RENDER_DPI = 200
MAX_IMAGES_PER_PAGE = 10


def load_pdf(pdf_bytes: bytes, source: str, doc_id: str, describer=None, min_chars: int = 40) -> list[Segment]:
    segments = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError(f"{source} is password-protected")

        for page_index, page in enumerate(doc):
            page_no = page_index + 1                         # humans count pages from 1
            text = page.get_text("text", sort=True).strip()
            scanned = False

            if len(text) >= min_chars:
                segments.append(Segment(text, source, doc_id, page_no, "text"))
            else:
                # Little or no text: probably a scanned page. Render it and OCR it.
                pix = page.get_pixmap(dpi=SCAN_RENDER_DPI)
                ocr_text = ocr_image(pix.tobytes("png"))
                if ocr_text:
                    scanned = True
                    segments.append(Segment(ocr_text, source, doc_id, page_no, "page_ocr"))
                elif text:
                    segments.append(Segment(text, source, doc_id, page_no, "text"))

            # A scanned page is one big image we already OCR'd; don't read it twice
            if not scanned:
                segments.extend(_page_images(doc, page, page_no, source, doc_id, describer))

    log.info("Loaded %s: %d segments", source, len(segments))
    return segments


def _page_images(doc, page, page_no, source, doc_id, describer) -> list[Segment]:
    out, seen = [], set()
    for img_index, info in enumerate(page.get_images(full=True)[:MAX_IMAGES_PER_PAGE]):
        xref = info[0]                                       # the image's ID inside the PDF
        if xref in seen:
            continue
        seen.add(xref)
        try:
            data = doc.extract_image(xref)
        except Exception as exc:
            log.debug("Skipping image %s on p.%s: %s", xref, page_no, exc)
            continue
        if data and data.get("image"):
            out.extend(image_to_segments(
                data["image"], source, doc_id, page=page_no, describer=describer,
                extra={"image_index": img_index},
            ))
    return out