import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Data
CSV_PATH = "2Melbourne_housing_FULL.csv"

if not os.path.exists(CSV_PATH):
    print(f"❌ Could not find '{CSV_PATH}' in this folder.")
    print("Put your CSV file here and update CSV_PATH.")
    exit()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("✅ Data loaded successfully!")

# Selecting columns

features = [
    'Rooms',
    'Distance',
    'Bathroom',
    'Car',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Regionname',
    'Type',
    'Propertycount'
]

target = 'Price'

df = df[features + [target]]
df = df.dropna(subset=[target])

# Cleaning data

numeric_cols = [
    'Rooms',
    'Distance',
    'Bathroom',
    'Car',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Propertycount'
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df['Regionname'] = df['Regionname'].fillna(df['Regionname'].mode()[0])
df['Type'] = df['Type'].fillna(df['Type'].mode()[0])

df = pd.get_dummies(
    df,
    columns=['Regionname', 'Type'],
    drop_first=True
)

print("✅ Data cleaned!")

# Splitting data

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training model

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("✅ Model training complete!")