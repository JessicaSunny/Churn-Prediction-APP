# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:33:46 2026

@author: JR
"""
import streamlit as st
from model import load_artifacts, predict_single

st.set_page_config(page_title="Churn Prediction Demo", layout="wide")
st.title("Churn Prediction Demo")

# Load the trained pipeline + metadata once at startup
try:
    pipe, metadata = load_artifacts()
except FileNotFoundError:
    st.error("No saved_model.pkl found. Run train.py first.")
    st.stop()

st.subheader("Enter user details")

input_dict = {}

# Numeric inputs — ranges/defaults pulled from metadata, not hardcoded
for col in metadata["numeric_cols"]:
    lo, hi, med = metadata["numeric_ranges"][col]
    input_dict[col] = st.number_input(
        col, min_value=float(lo), max_value=float(hi), value=float(med)
    )

# Categorical inputs — options pulled from metadata, not hardcoded
for col in metadata["categorical_cols"]:
    options = metadata["categorical_options"][col]
    input_dict[col] = st.selectbox(col, options)

st.subheader("Prediction")

if st.button("Predict"):
    pred, proba = predict_single(pipe, metadata, input_dict)
    st.metric("Churn probability", f"{proba:.1%}")
    if pred == 1:
        st.warning("Predicted: likely to churn")
    else:
        st.success("Predicted: likely to stay")