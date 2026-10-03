from bers.normalize.text_utils import clean_text

# Common corporate suffixes
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'l l c', 'plc', 'ltd', 'limited',
    'co', 'company', 'lp', 'llp', 'gmbh', 'sa', 'nv'
}

def normalize_name(name: str) -> str:
    """
    Normalizes business names, handling DBA (doing business as) 
    and stripping common legal suffixes.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    
    # Handle DBA: Often the DBA name is the most distinctive for matching.
    # We take the part after 'dba' if it exists.
    if ' dba ' in cleaned:
        parts = cleaned.split(' dba ')
        cleaned = parts[-1].strip()
    elif ' doing business as ' in cleaned:
        parts = cleaned.split(' doing business as ')
        cleaned = parts[-1].strip()
        
    # Strip legal suffixes from the end of the name
    words = cleaned.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
        
    return " ".join(words)
