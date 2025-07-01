import numpy as np

def assert_minimum_precision(precision, threshold=0.85):
    assert precision >= threshold, f"Precision {precision:.2f} below threshold {threshold:.2f}"

def check_drift(old_preds, new_preds, tolerance=0.05):
    drift = np.abs(np.mean(new_preds) - np.mean(old_preds))
    assert drift <= tolerance, f"Drift {drift:.2f} exceeds tolerance {tolerance:.2f}"
