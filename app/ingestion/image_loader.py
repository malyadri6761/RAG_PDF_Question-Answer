import io
import logging
import os

import pytesseract
from PIL import Image, ImageOps

from . import Segment

log = logging.getLogger(__name__)

MIN_IMAGE_SIDE = 64      # smaller than this = icon/logo, skip it
OCR_MAX_SIDE = 3000      # shrink huge images before OCR

# --- Windows: find tesseract.exe even if it's not on PATH ---
_tess = os.getenv("TESSERACT_CMD", "").strip()
if not _tess and os.name == "nt":                        # "nt" means Windows
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default):
        _tess = default
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess


def prepare_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)                   # fix phone-photo rotation
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")                         # handle PNG transparency, CMYK, etc.
    longest = max(img.size)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    elif longest < 1000:
        scale = 1000 / longest                           # upscale: small text OCRs badly
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.grayscale(img)


def clean_ocr(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if sum(ch.isalnum() for ch in line) >= 2]   # drop "|", "—", ...
    return "\n".join(lines).strip()


def ocr_image(image_bytes: bytes) -> str:
    """Read the text in an image. Returns '' on failure instead of crashing."""
    try:
        img = prepare_for_ocr(Image.open(io.BytesIO(image_bytes)))
        return clean_ocr(pytesseract.image_to_string(img))
    except Exception as exc:
        log.warning("OCR failed: %s", exc)
        return ""


def is_meaningful_image(image_bytes: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return min(img.size) >= MIN_IMAGE_SIDE
    except Exception:
        return False                                     # not a readable image


def image_to_segments(image_bytes, source, doc_id, page=None, describer=None, extra=None):
    """One image -> up to two Segments: a caption (if describer given) and OCR text."""
    if not is_meaningful_image(image_bytes):
        return []
    extra = dict(extra or {})
    segments = []

    if describer is not None:
        try:
            caption = (describer(image_bytes) or "").strip()
        except Exception as exc:
            log.warning("Captioning failed for %s: %s", source, exc)
            caption = ""
        if caption:
            segments.append(Segment(caption, source, doc_id, page, "image_caption", dict(extra)))

    text = ocr_image(image_bytes)
    if text:
        segments.append(Segment(text, source, doc_id, page, "image_ocr", dict(extra)))
    return segments