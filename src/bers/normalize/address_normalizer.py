from bers.normalize.text_utils import clean_text

ADDRESS_ABBR = {
    'street': 'st',
    'avenue': 'ave',
    'boulevard': 'blvd',
    'road': 'rd',
    'drive': 'dr',
    'lane': 'ln',
    'court': 'ct',
    'suite': 'ste',
    'apartment': 'apt',
    'room': 'rm',
    'building': 'bldg',
    'floor': 'fl',
    'north': 'n',
    'south': 's',
    'east': 'e',
    'west': 'w'
}

def normalize_address(address: str) -> str:
    """
    Normalizes an address line by standardizing common abbreviations.
    """
    cleaned = clean_text(address)
    if not cleaned:
        return ""
    
    words = cleaned.split()
    normalized_words = [ADDRESS_ABBR.get(w, w) for w in words]
    return " ".join(normalized_words)
