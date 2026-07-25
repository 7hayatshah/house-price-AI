# Melbourne Housing Price Prediction

## Project Overview
This project is an ML application that predicts house prices in Melbourne city using machine learning. Users can enter house details through a graphical interface and receive a predicted house price along with an explanation of the prediction.

## Features
- Predicts house prices using a trained machine learning models
- User-friendly graphical interface
- Displays prediction results
- Shows model evaluation metrics
- Explains prediction using feature importance (if applicable)

## Dataset
- Dataset: Melbourne Housing Dataset
- Source: Kaggle
- Target Variable: Price

## Technologies Used
- Python 3.x — core language
- Streamlit — the UI/web app
- Pandas — data loading, cleaning, manipulation
- NumPy — numerical operations (used in RMSE calc, array handling)
- Scikit-learn — Linear Regression, Random Forest, train/test split, evaluation metrics
- Matplotlib — the bar charts for feature contributions/SHAP visualization
- Joblib — saving/loading trained models to .joblib files
- SHAP — model interpretability (explaining Random Forest predictions)

## Project Structure


Project/
│── ui.py
│── requirements.txt
│── README.md
│── data/
│   └── Melbourne_housing.csv
│── screenshots/
    └── assets


## Installation

1. Clone the repository

bash
git clone <repository-link>


2. Open the project folder


3. Install dependencies (if first time)

bash
pip install -r requirements.txt


4. Run the application

For Streamlit:

bash
streamlit run app.py


## How to Use

1. Launch the application.
2. Enter house details.
3. Click the Predict button.
4. View the predicted house price.
5. Review the evaluation metrics and explanation.

## Machine Learning Model

- Algorithm: (Random Forest Regressor/ Linear Regression)
- Target: House Price
- Features: Rooms, Bathroom, Car, Landsize, YearBuilt, Region, etc.

## Evaluation

Example:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score