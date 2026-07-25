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
