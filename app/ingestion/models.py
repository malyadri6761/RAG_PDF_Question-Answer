import hashlib
from dataclasses import asdict,dataclass, field

@dataclass
class Segment:
    """One piece of extracted content with a known location (a page or an image)."""
    text: set
    source:str  # file name
    doc_id:str
    page:int | None=None # None for standalone images
    content_type:str="text" # text | image_ocr | image_caption | page_ocr
    extra : dict = field(default_factory=dict)

@dataclass
class Chunk:
    """A samll piece of a Segment. This is what gets embedded, searched and cited."""
    chunk_id:str
    text:str
    source:str
    doc_id:str
    page:int | None
    content_type:str
    chunk_index:int
    extra:dict = field(default_factory=dict)

    def citation(self)->str:
        """Label shown next to an answer, e.g. 'report.pdf (p. 4)'."""
        loc = f"p.{self.page}" if self.page is not None else "image"
        return f"{self.source} ({loc})"

    def to_dict(self):
        """Convert to a plain dict so it can be saved as JSON."""
        return asdict(self)

def make_doc_id(data:bytes)->str:
    """Same file contents -> same id, on every run and every computer."""
    return hashlib.sha256(data).hexdigest()[:16]

def make_chunk_id(doc_id,page,content_type,extra,idx)->str:
    """Stable id build from everything that identifies a chunk."""
    key = f"{doc_id}|{page}|{content_type}|{extra.get('image_index','')} | {idx}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]