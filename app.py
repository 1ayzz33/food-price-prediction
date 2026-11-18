from flask import Flask, render_template, request, flash, redirect, url_for
import joblib

app = Flask(__name__)
app.secret_key = "food-price-prediction-secret-key"

# Load model and encoders at startup
model = joblib.load("models/model.pkl")
commodity_encoder = joblib.load("models/commodity_encoder.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

# Valid options from the trained encoders
VALID_COMMODITIES = list(commodity_encoder.classes_)
VALID_STATES = list(state_encoder.classes_)


@app.route("/")
def home():
    return render_template(
        "index.html",
        commodities=VALID_COMMODITIES,
        states=VALID_STATES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    errors = []

    # --- Collect and validate inputs ---
    commodity = (request.form.get("commodity") or "").strip()
    state = (request.form.get("state") or "").strip()
    month_raw = (request.form.get("month") or "").strip()
    fuel_raw = (request.form.get("fuel") or "").strip()
    inflation_raw = (request.form.get("inflation") or "").strip()
    rainfall_raw = (request.form.get("rainfall") or "").strip()

    if not commodity:
        errors.append("Please select a commodity.")
    elif commodity not in VALID_COMMODITIES:
        errors.append("Invalid commodity selected.")

    if not state:
        errors.append("Please select a state.")
    elif state not in VALID_STATES:
        errors.append("Invalid state selected.")

    try:
        month = int(month_raw)
        if month < 1 or month > 12:
            errors.append("Month must be between 1 and 12.")
    except (TypeError, ValueError):
        errors.append("Please enter a valid month (1-12).")
        month = None

    try:
        fuel = float(fuel_raw)
        if fuel <= 0:
            errors.append("Fuel price must be greater than zero.")
    except (TypeError, ValueError):
        errors.append("Please enter a valid fuel price.")
        fuel = None

    try:
        inflation = float(inflation_raw)
        if inflation < 0:
            errors.append("Inflation rate cannot be negative.")
    except (TypeError, ValueError):
        errors.append("Please enter a valid inflation rate.")
        inflation = None

    try:
        rainfall = float(rainfall_raw)
        if rainfall < 0:
            errors.append("Rainfall cannot be negative.")
    except (TypeError, ValueError):
        errors.append("Please enter a valid rainfall value.")
        rainfall = None

    if errors:
        for error in errors:
            flash(error, "error")
        return redirect(url_for("home"))

    # --- Encode categorical inputs ---
    try:
        commodity_enc = commodity_encoder.transform([commodity])[0]
        state_enc = state_encoder.transform([state])[0]
    except ValueError as exc:
        flash(f"Encoding error: {exc}", "error")
        return redirect(url_for("home"))

    # --- Predict ---
    try:
        result = model.predict(
            [[commodity_enc, state_enc, month, fuel, inflation, rainfall]]
        )
        predicted_price = round(float(result[0]), 2)
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("home"))

    return render_template(
        "result.html",
        price=predicted_price,
        commodity=commodity,
        state=state,
    )


if __name__ == "__main__":
    app.run(debug=True)
