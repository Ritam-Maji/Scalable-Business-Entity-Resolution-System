from bers.normalize.text_utils import clean_text
from bers.normalize.name_normalizer import normalize_name
from bers.normalize.address_normalizer import normalize_address
from bers.normalize.country_utils import normalize_country

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("Café") == "cafe"
    assert clean_text("  Extra   Spaces  ") == "extra spaces"
    assert clean_text(None) == ""

def test_normalize_name():
    assert normalize_name("Acme Corp LLC") == "acme"
    assert normalize_name("Widgets Incorporated") == "widgets"
    assert normalize_name("Big Business DBA Small Shop") == "small shop"
    
def test_normalize_address():
    assert normalize_address("123 Main Street Suite 4") == "123 main st ste 4"
    assert normalize_address("456 North Avenue") == "456 n ave"
    
def test_normalize_country():
    assert normalize_country("United States") == "us"
    assert normalize_country("U.S.A.") == "us"
    assert normalize_country("UnknownLand") == "unknownland"
