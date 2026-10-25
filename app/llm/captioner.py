from .client import get_llm_client

CAPTION_SYSTEM = (
    "You describe images so they can be found by text search and used to answer questions. "
    "Be factual and specific. Never invent details that are not visible."
)

CAPTION_PROMPT = """Describe this image for a search index.
- Start with what kind of image it is (chart, table, diagram, photo, screenshot, scanned text).
- For charts and tables: the title, axes or columns, and the key values and trends.
- For diagrams: the components and how they connect.
- Copy any important visible text exactly.
Keep it under 150 words. Plain text, no markdown."""


class ImageCaptioner:
    def __init__(self, llm=None):
        self.llm = llm or get_llm_client()

    def __call__(self, image_bytes: bytes) -> str:
        return self.llm.generate(CAPTION_SYSTEM, CAPTION_PROMPT, images=[image_bytes])