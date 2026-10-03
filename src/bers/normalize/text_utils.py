import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Basic text normalization: lowercase, strip accents, remove special chars.
    Returns an empty string if input is not a string (e.g., NaN).
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Strip accents (NFD decomposition)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation (replace with space to avoid joining words)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
