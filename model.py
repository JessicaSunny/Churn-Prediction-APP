# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:33:49 2026

@author: JR
"""

import joblib
import json
import pandas as pd


def load_artifacts(model_path="saved_model.pkl", metadata_path="metrics.json"):
    """Load the fitted pipeline and its metadata (column names, ranges, metrics)."""
    pipe = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return pipe, metadata


def predict_single(pipe, metadata, input_dict: dict):
    """
    input_dict: {feature_name: value} for every raw feature the model expects
    (e.g. {"tenure_days": 45, "is_verified": 0, "account_type": "standard", ...})
    Returns (predicted_label, predicted_probability).
    """
    cols = metadata["numeric_cols"] + metadata["categorical_cols"]
    row = pd.DataFrame([{c: input_dict.get(c) for c in cols}])
    proba = pipe.predict_proba(row)[0, 1]
    pred = int(proba >= 0.5)
    return pred, float(proba)


def predict_batch(pipe, metadata, df: pd.DataFrame):
    """Score a whole dataframe at once (e.g. an uploaded CSV of many users)."""
    cols = metadata["numeric_cols"] + metadata["categorical_cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing expected columns: {missing}")
    proba = pipe.predict_proba(df[cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["predicted_proba"] = proba
    out["predicted_label"] = pred
    return out