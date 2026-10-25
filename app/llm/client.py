import base64
import io
import os
import re

from PIL import Image

MAX_IMAGE_SIDE = 1568      # bigger images cost more tokens without helping


class LLMNotConfigured(RuntimeError):
    """Missing API key, unknown provider, or SDK not installed."""


def to_png(image_bytes: bytes) -> bytes:
    """Any image -> modest-size RGB PNG."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))      # shrinks only if larger, keeps aspect ratio
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMNotConfigured("ANTHROPIC_API_KEY is not set in .env")
        self.client = anthropic.Anthropic()               # reads the key from the environment
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, prompt: str, images: list[bytes] | None = None) -> str:
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(to_png(img))}}
            for img in images or []
        ]
        content.append({"type": "text", "text": prompt})
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


class OpenAIClient:
    def __init__(self, model: str, max_tokens: int = 1024):
        import openai
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMNotConfigured("OPENAI_API_KEY is not set in .env")
        self.client = openai.OpenAI()
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, prompt: str, images: list[bytes] | None = None) -> str:
        content = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(to_png(img))}"}}
            for img in images or []
        ]
        resp = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiClient:
    def __init__(self, model: str, max_tokens: int = 1024):
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMNotConfigured("GEMINI_API_KEY is not set in .env")
        self.client = genai.Client(api_key=key)
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, prompt: str, images: list[bytes] | None = None) -> str:
        from google.genai import types
        parts = [types.Part.from_bytes(data=to_png(img), mime_type="image/png") for img in images or []]
        resp = self.client.models.generate_content(
            model=self.model, contents=[*parts, prompt],
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=self.max_tokens),
        )
        return (resp.text or "").strip()


class GroqClient:
    MAX_IMAGES = 3                                        # Groq's per-request limit for qwen/qwen3.8-27b
    _THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(self, model: str, max_tokens: int = 1024):
        import groq
        if not os.getenv("GROQ_API_KEY"):
            raise LLMNotConfigured("GROQ_API_KEY is not set in .env")
        self.client = groq.Groq()                         # reads GROQ_API_KEY from the environment
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, prompt: str, images: list[bytes] | None = None) -> str:
        content = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(to_png(img))}"}}
            for img in (images or [])[: self.MAX_IMAGES]
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        )
        text = resp.choices[0].message.content or ""
        return self._THINK.sub("", text).strip()          # drop any <think>...</think> reasoning
PROVIDERS = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
}
PACKAGES = {"anthropic": "anthropic", "openai": "openai", "gemini": "google-genai", "groq": "groq"}


def get_llm_client():
    """Build the client chosen in .env (LLM_PROVIDER + LLM_MODEL)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if provider not in PROVIDERS:
        raise LLMNotConfigured(f"Unknown LLM_PROVIDER '{provider}'. Use: {', '.join(PROVIDERS)}")
    if not model:
        raise LLMNotConfigured("Set LLM_MODEL in .env")
    try:
        return PROVIDERS[provider](model)
    except ImportError as exc:
        raise LLMNotConfigured(f"Run: uv add {PACKAGES[provider]}") from exc