import pandas as pd
from bers.blocking.exact_key_blocker import ExactKeyBlocker
from bers.blocking.token_blocker import TokenBlocker
from bers.constants import COL_RECORD_ID

def test_exact_key_blocker():
    df = pd.DataFrame({
        COL_RECORD_ID: ['1', '2', '3', '4'],
        'city': ['A', 'A', 'B', 'B'],
        'country': ['US', 'US', 'US', 'CA']
    })
    blocker = ExactKeyBlocker(blocking_keys=['city', 'country'])
    pairs = blocker.block(df)
    # 1 and 2 share city A, country US.
    # 3 and 4 have different countries.
    assert pairs == {('1', '2')}

def test_token_blocker():
    df = pd.DataFrame({
        COL_RECORD_ID: ['1', '2', '3'],
        'name': ['acme corp', 'acme inc', 'widgets llc']
    })
    blocker = TokenBlocker(target_column='name')
    pairs = blocker.block(df)
    # 'acme' is shared by 1 and 2.
    assert pairs == {('1', '2')}
