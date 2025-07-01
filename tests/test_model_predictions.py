import pandas as pd
from sklearn.metrics import precision_score
from ml_model_testing import metrics, validators, visualizer

def test_model_output_consistency():
    df_v1 = pd.read_csv("tests/sample_data/model_v1_predictions.csv")
    df_v2 = pd.read_csv("tests/sample_data/model_v2_predictions.csv")

    y_true = df_v1["label"]
    y_pred_v1 = df_v1["prediction"]
    y_pred_v2 = df_v2["prediction"]

    validators.check_drift(y_pred_v1, y_pred_v2, tolerance=0.05)
    metrics.evaluate_classification(y_true, y_pred_v2)
    precision = precision_score(y_true, y_pred_v2, average="binary")
    validators.assert_minimum_precision(precision)
