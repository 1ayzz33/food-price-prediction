import joblib

# Load model and encoders
model = joblib.load("models/model.pkl")
commodity_encoder = joblib.load("models/commodity_encoder.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

# Convert text into numbers
commodity = commodity_encoder.transform(["Rice"])[0]
state = state_encoder.transform(["Kano"])[0]

# Predict
prediction = model.predict([
    [commodity, state, 6, 950, 30, 80]
])

print(f"Predicted price for Rice in Kano (month 6): NGN {prediction[0]:,.2f}")
