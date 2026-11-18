import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load data
df = pd.read_csv("food_price.csv")

# Convert text to numbers
commodity_encoder = LabelEncoder()
state_encoder = LabelEncoder()

df["Commodity"] = commodity_encoder.fit_transform(df["Commodity"])
df["State"] = state_encoder.fit_transform(df["State"])

# Input variables
X = df.drop("Price", axis=1)

# Output variable
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"  MAE  (Mean Absolute Error): NGN {mae:,.2f}")
print(f"  RMSE (Root Mean Squared Error): NGN {rmse:,.2f}")
print(f"  R²  Score: {r2:.4f}")

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(commodity_encoder, "models/commodity_encoder.pkl")
joblib.dump(state_encoder, "models/state_encoder.pkl")

print("\nModel saved successfully to models/")
