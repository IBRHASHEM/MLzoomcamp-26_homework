
import pickle
import math
import re

from flask import Flask, request, jsonify, render_template

# --------------------------------------------------
# 1. Load trained model package
# --------------------------------------------------

with open("stroke_prediction_model.bin", "rb") as f_in:
    package = pickle.load(f_in)

model = package["model"]
dv = package["dv"]
threshold = package["threshold"]

numerical_medians = package["numerical_medians"]
categorical_modes = package["categorical_modes"]

numerical_features = package["numerical_features"]
categorical_features = package["categorical_features"]
feature_columns = package["feature_columns"]

bmi_upper_cap = package["bmi_upper_cap"]


# --------------------------------------------------
# 2. Initialize Flask application
# --------------------------------------------------

app = Flask("stroke-prediction")

# Allow the GitHub Pages frontend to call this API from the browser.
ALLOWED_ORIGINS = {"https://ibrhashem.github.io"}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")

    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    return response

# --------------------------------------------------
# 3. Web Interface and Health Check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": package["model_name"]
    }), 200

# --------------------------------------------------
# 3. Preprocess incoming patient data
# --------------------------------------------------

def preprocess_patient(patient):

    if not isinstance(patient, dict):
        raise ValueError("Patient data must be a JSON object.")

    unknown_fields = set(patient.keys()) - set(feature_columns)

    if unknown_fields:
        raise ValueError(
            f"Unknown fields: {sorted(unknown_fields)}"
        )

    processed = {}

    # Process numerical features.
    for column in numerical_features:

        value = patient.get(column)

        if value is None or value == "":
            value = numerical_medians[column]
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid numerical value for '{column}'."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"'{column}' must be a finite number."
                )

        processed[column] = value

    # Apply the same BMI upper cap used during training.
    processed["bmi"] = min(
        processed["bmi"],
        bmi_upper_cap
    )

    # Process categorical features.
    for column in categorical_features:

        value = patient.get(column)

        if value is None or value == "":
            value = categorical_modes[column]
        else:
            value = str(value).strip().lower()
            value = re.sub(r"\s+", "_", value)

        processed[column] = value

    return processed

# --------------------------------------------------
# Web Form Prediction
# --------------------------------------------------

@app.route("/predict-form", methods=["POST"])
def predict_form():

    patient = request.form.to_dict()

    try:
        processed_patient = preprocess_patient(patient)

        X = dv.transform([processed_patient])

        probability = float(
            model.predict_proba(X)[0, 1]
        )

        prediction = int(probability >= threshold)

        result = {
            "stroke_probability": probability,
            "stroke_prediction": prediction,
            "threshold": threshold,
            "model": package["model_name"],
        }

        return render_template(
            "index.html",
            result=result
        )

    except ValueError as error:
        return render_template(
            "index.html",
            error=str(error)
        ), 400
# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    patient = request.get_json(silent=True)

    if patient is None:
        return jsonify({
            "error": "Request body must contain valid JSON."
        }), 400

    try:
        processed_patient = preprocess_patient(patient)

        # Transform patient data using the trained vectorizer.
        X = dv.transform([processed_patient])

        # Predict stroke probability.
        probability = float(
            model.predict_proba(X)[0, 1]
        )

        # Apply the selected decision threshold.
        prediction = int(probability >= threshold)

        result = {
            "stroke_probability": round(probability, 4),
            "stroke_prediction": prediction,
            "threshold": threshold,
            "model": package["model_name"],
        }

        return jsonify(result)

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


# --------------------------------------------------
# 5. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9696,
        debug=False
    )
