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

#  3: Selecting columns 
features = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize',
            'BuildingArea', 'YearBuilt', 'Regionname',
            'Type', 'Propertycount']
target = 'Price'

df = df[features + [target]]
df = df.dropna(subset=[target])

print("✅ Data loaded successfully!")

#  4: Cleaning data 
numeric_cols = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize',
                 'BuildingArea', 'YearBuilt', 'Propertycount']

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df['Regionname'] = df['Regionname'].fillna(df['Regionname'].mode()[0])
df['Type'] = df['Type'].fillna(df['Type'].mode()[0])

df = pd.get_dummies(df, columns=['Regionname', 'Type'], drop_first=True)

print("✅ Data cleaned!")

#  5: Splitting target 
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  6: Training model 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("✅ Model training complete!")
#  7: Prediction function 
def predict_price(input_dict, model, reference_columns):
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, columns=['Regionname', 'Type'], drop_first=True)

    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[reference_columns]
    prediction = model.predict(input_df)[0]
    return prediction
    #  8: Taking input 
print("\nEnter house details:")
user_input = {
    'Rooms': int(input("Rooms: ")),
    'Distance': float(input("Distance from CBD (km): ")),
    'Bathroom': int(input("Bathrooms: ")),
    'Car': int(input("Car spots: ")),
    'Landsize': float(input("Landsize (sqm): ")),
    'BuildingArea': float(input("Building Area (sqm): ")),
    'YearBuilt': int(input("Year Built: ")),
    'Propertycount': int(input("Property count in suburb: ")),
    'Regionname': input("Region name (e.g. Northern Metropolitan): "),
    'Type': input("Type (h/u/t): ")
}
#  9: Showing result 
predicted_price = predict_price(user_input, rf, X.columns)
print(f"\n💰 Predicted Price: ${predicted_price:,.0f}")
