import re
import unicodedata


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text


def remove_hidden_chars(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\u200b", "", text)
    return text


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_hidden_chars(text)
    text = clean_whitespace(text)
    return text


def extract_metadata_from_content(text: str) -> dict:
    metadata = {}
    lines = text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        if len(first_line) < 200:
            metadata["title"] = first_line[len("# "):].strip() if first_line.startswith("# ") else first_line
    text_lower = text.lower()
    date_patterns = [
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            metadata["date_found"] = match.group(1)
            break
    return metadata
