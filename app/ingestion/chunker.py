import re
from .models import Chunk,Segment,make_chunk_id

# Preferred cut points, best first
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

def normalize(text:str)->str:
    """Clean up messy PDF/OCR text before splitting."""
    text = text.replace("\r\n","\n").replace("\u00a0"," ")  # Windows newlines, non-breaking space
    text = re.sub(r"-\n(?=[a-z])","",text)                  # "retri-\neval"->"retrieval"
    text = re.sub(r"[ \t]+"," ",text)                       # many spaces/tabs -> one space
    text = re.sub(r"\n{3,}","\n\n",text)                    # many blank lines -> one
    return text.strip()

def best_cut(window:str,min_pos:int)->int:
    """Where to end this chunk : the last good separator after min_pos."""
    for sep in SEPARATORS:
        pos = window.rfind(sep) # rfind = search from the END of the window
        if pos >= min_pos:
            return pos+len(sep) # cut just after the seperator
    return len(window)          # no good break: hard cut at the limit


def split_text(text:str, chunk_size:int=800,overlap:int=120)->list[str]:
    if overlap>= chunk_size:
        return ValueError("Overlap must be smaller than chunk_size.")

    text = normalize(text)
    if len(text)<=chunk_size:    # short text : one chunk ( or none if empty)
        return [text] if text else []

    pieces = []
    start,n = 0,len(text)

    while start < n:
        end = min(start+chunk_size,n)
        if end<n:                # not the last piece : look for a nice cut point
            end = start+best_cut(text[start:end],min_pos=chunk_size//2)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end>=n:
            break

        # Start the next chunk 'overlap' characters back, then move to the next word
        next_start = max(end-overlap,start+1) # +1 guarantess we always move forward
        space = text.find(" ",next_start,end)
        start = space+1 if space !=-1 else next_start
    return pieces


def chunk_segments(segments:list[Segment],chunk_size:int=800,overlap:int=120)->list[Chunk]:
    """Split every segment separately, so a chunk never spans two pages."""
    chunks = []
    for seg in segments:
        for i, piece in enumerate(split_text(seg.text,chunk_size,overlap)):
            chunks.append(Chunk(
                chunk_id = make_chunk_id(seg.doc_id,seg.page,seg.content_type,seg.extra,i),
                text=piece,
                source=seg.source,
                doc_id=seg.doc_id,
                page=seg.page,                  # page number copied from the  segment
                content_type=seg.content_type,
                chunk_index=i,                  
                extra=dict(seg.extra),          # a copy, not shared

            ))
    return chunks