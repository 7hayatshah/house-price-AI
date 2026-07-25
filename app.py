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

with st.sidebar:

    st.header("📈 Model Performance")

    st.subheader("Linear Regression")
    st.write(f"RMSE: ${lr_metrics.get('rmse',0):,.0f}")
    st.write(f"MAE: ${lr_metrics.get('mae',0):,.0f}")
    st.write(f"R²: {lr_metrics.get('r2',0):.3f}")

    st.subheader("Random Forest")
    st.write(f"RMSE: ${rf_metrics.get('rmse',0):,.0f}")
    st.write(f"MAE: ${rf_metrics.get('mae',0):,.0f}")
    st.write(f"R²: {rf_metrics.get('r2',0):.3f}")

st.title("🏠 Melbourne House Price Predictor")

st.write(
    "Enter house details below to predict the house price."
)

col1, col2 = st.columns(2)

with col1:
    rooms = st.number_input("Rooms", 1, 10, 3)
    distance = st.number_input("Distance", 0.0, value=10.0)
    bathroom = st.number_input("Bathrooms", 1, 5, 2)
    car = st.number_input("Car Spots", 0, 5, 1)
    landsize = st.number_input("Landsize", 0.0, value=450.0)

with col2:
    building_area = st.number_input("Building Area", 0.0, value=150.0)
    year_built = st.number_input("Year Built", 1900, 2024, 2005)
    propertycount = st.number_input("Property Count", 0, value=4000)

    regionname = st.selectbox(
        "Region",
        [
            "Northern Metropolitan",
            "Southern Metropolitan",
            "Western Metropolitan",
            "Eastern Metropolitan",
            "South-Eastern Metropolitan",
            "Northern Victoria",
            "Eastern Victoria",
            "Western Victoria",
        ],
    )

    type_options = {
        "House": "h",
        "Unit/Apartment": "u",
        "Townhouse": "t",
    }

    house_type = type_options[
        st.selectbox("Property Type", list(type_options.keys()))
    ]

st.divider()




    if st.button("🔍 Predict Price", use_container_width=True):

    input_dict = {
        "Rooms": rooms,
        "Distance": distance,
        "Bathroom": bathroom,
        "Car": car,
        "Landsize": landsize,
        "BuildingArea": building_area,
        "YearBuilt": year_built,
        "Propertycount": propertycount,
        "Regionname": regionname,
        "Type": house_type,
    }

    lr_prediction = predict_price(
        input_dict,
        lr,
        lr_columns,
    )

    rf_prediction = predict_price(
        input_dict,
        rf,
        rf_columns,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "📊 Linear Regression",
            f"${lr_prediction:,.0f}",
        )

    with c2:
        st.metric(
            "🌲 Random Forest",
            f"${rf_prediction:,.0f}",




            st.divider()
    st.subheader("🔎 Model Explanation")

    explain1, explain2 = st.columns(2)

    with explain1:

        st.markdown("### Linear Regression")

        lr_input_df = build_input_df(input_dict, lr_columns)

        contributions = (
            lr_input_df.iloc[0].values * lr.coef_
        )

        contribution = pd.Series(
            contributions,
            index=lr_columns,
        )

        contribution = contribution.reindex(
            contribution.abs().sort_values(
                ascending=False
            ).index
        ).head(10)

        fig, ax = plt.subplots(figsize=(6,5))

        colors = [
            "green" if x > 0 else "red"
            for x in contribution.values
        ]

        ax.barh(
            contribution.index[::-1],
            contribution.values[::-1],
            color=colors[::-1],
        )

        st.pyplot(fig)

    with explain2:

        st.markdown("### Random Forest (SHAP)")

        rf_input_df = build_input_df(
            input_dict,
            rf_columns,
        )

        shap_values = rf_explainer.shap_values(
            rf_input_df
        )

        shap_series = pd.Series(
            shap_values[0],
            index=rf_columns,
        )

        shap_series = shap_series.reindex(
            shap_series.abs().sort_values(
                ascending=False
            ).index
        ).head(10)

        fig2, ax2 = plt.subplots(figsize=(6,5))

        colors = [
            "green" if x > 0 else "red"
            for x in shap_series.values
        ]

        ax2.barh(
            shap_series.index[::-1],
            shap_series.values[::-1],
            color=colors[::-1],
        )

        st.pyplot(fig2)
        )
