#  1: Importing dependencies
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#  2: Load Data
CSV_PATH = "2Melbourne_housing_FULL.csv" 

if not os.path.exists(CSV_PATH):
    print(f"❌ Could not find '{CSV_PATH}' in this folder.")
    print("   Put your CSV file here and update CSV_PATH at the top of this script,")
    print("   or provide the full path, e.g. CSV_PATH = r'C:\\Users\\You\\Desktop\\data.csv'")
    exit()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("✅ Data loaded successfully!")
