from bers.features.name_features import extract_name_features
from bers.features.agreement_features import extract_agreement_features

def test_extract_name_features():
    feat = extract_name_features("acme", "acme")
    assert feat['name_exact_match'] == 1.0
    assert feat['name_jaccard'] == 1.0
    assert feat['name_seq_matcher'] == 1.0
    
    feat = extract_name_features("acme", "widgets")
    assert feat['name_exact_match'] == 0.0
    assert feat['name_jaccard'] == 0.0
    assert 0.0 <= feat['name_seq_matcher'] <= 1.0

def test_extract_agreement_features():
    feat = extract_agreement_features("A", "A", "B", "C", "", "US", "123", "123")
    assert feat['city_match'] == 1.0
    assert feat['state_match'] == 0.0
    assert feat['country_match'] == -1.0 # explicitly testing missing data indicator
    assert feat['zip_match'] == 1.0
