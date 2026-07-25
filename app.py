import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model(save_path):
    data = joblib.load(save_path)
    return data["model"], data["columns"], data.get("metrics", {})

lr, lr_columns, lr_metrics = load_model("linear_regression_model.joblib")
rf, rf_columns, rf_metrics = load_model("random_forest_model.joblib")

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

rf_explainer = get_shap_explainer(rf)

def build_input_df(input_dict, reference_columns):
    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(
        input_df,
        columns=["Regionname", "Type"],
        drop_first=True
    )

    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[reference_columns]


def predict_price(input_dict, model, reference_columns):
    input_df = build_input_df(input_dict, reference_columns)
    return model.predict(input_df)[0
