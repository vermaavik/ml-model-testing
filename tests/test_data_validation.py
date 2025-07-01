import pandas as pd

def test_feature_ranges():
    df = pd.read_csv("tests/sample_data/test_features.csv")

    assert df["age"].between(0, 100).all(), "Invalid age values"
    assert df["income"].between(0, 1_000_000).all(), "Income out of range"
