"""Text normalization and lightweight tokenization for FCE GEC."""
from __future__ import annotations
import re
import unicodedata

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"'’”)]*)")


def normalize_text(text: str) -> str:
    """Normalize Unicode/newlines/whitespace without removing case or punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize words and punctuation. Case is intentionally preserved for GEC."""
    return TOKEN_RE.findall(text)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence spans while preserving character offsets."""
    spans=[]
    start=0
    for m in SENTENCE_END_RE.finditer(text):
        end=m.end()
        # Require sentence boundary before whitespace/newline/end.
        if end == len(text) or text[end].isspace():
            chunk=text[start:end].strip()
            if chunk:
                left=start
                while left < end and text[left].isspace(): left += 1
                spans.append((left,end))
            start=end
    if start < len(text):
        left=start
        while left < len(text) and text[left].isspace(): left += 1
        if left < len(text): spans.append((left,len(text)))
    return spans
