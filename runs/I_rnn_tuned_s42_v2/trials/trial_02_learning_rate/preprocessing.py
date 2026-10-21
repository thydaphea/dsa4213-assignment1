import re
import unicodedata

TOKEN_PATTERN = r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.,]\d+)*|[^\w\s]|_"
TOKEN_RE = re.compile(TOKEN_PATTERN, flags=re.UNICODE)

def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('’', "'").replace('‘', "'")
    return ' '.join(text.lower().split())

def tokenize(text):
    return TOKEN_RE.findall(normalize_text(text))
