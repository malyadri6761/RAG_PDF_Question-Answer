import io
import re
import shutil
import zlib
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont




def _has_tesseract() -> bool:
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return shutil.which("tesseract") is not None or (os.name == "nt" and os.path.exists(default))

needs_tesseract = pytest.mark.skipif(not _has_tesseract(), reason="Tesseract not installed")



def make_text_image(lines, size=(1200, 400), font_size=48) -> bytes:
    """A white PNG with black text, for OCR tests."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)          # Windows
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)  # Linux
        except OSError:
            font = ImageFont.load_default(size=font_size)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += int(font_size * 1.6)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words score higher. No model download."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Returns a fixed reply and records every call. No API key needed."""

    def __init__(self, reply="The answer is in the first source [1]."):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt, images=None):
        self.calls.append({"system": system, "prompt": prompt, "images": images or []})
        return self.reply