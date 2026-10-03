from bers.normalize.text_utils import clean_text

# A simple open-set mapping for common countries to their 2-letter ISO codes.
# This can be expanded via configs or an external library if needed.
COUNTRY_MAPPING = {
    'united states': 'us',
    'united states of america': 'us',
    'usa': 'us',
    'u s a': 'us',
    'us': 'us',
    'united kingdom': 'gb',
    'uk': 'gb',
    'u k': 'gb',
    'great britain': 'gb',
    'canada': 'ca',
    'germany': 'de',
    'deutschland': 'de',
    'france': 'fr',
    'australia': 'au',
    'india': 'in',
    'china': 'cn',
    'japan': 'jp',
    'brazil': 'br',
    'mexico': 'mx'
}

def normalize_country(country: str) -> str:
    """
    Normalizes country names to a standard 2-letter code where possible.
    For unmapped ('open-set') countries, returns the cleaned string.
    """
    cleaned = clean_text(country)
    if not cleaned:
        return ""
    
    return COUNTRY_MAPPING.get(cleaned, cleaned)
