import pickle

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# Load the saved pipeline artifact
with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# Handle both scikit-learn Pipeline objects and (dv, model) tuples
if hasattr(pipeline, "predict_proba"):
    y_pred = pipeline.predict_proba([lead])[0, 1]
else:
    dv, model = pipeline
    X = dv.transform([lead])
    y_pred = model.predict_proba(X)[0, 1]

print(f"Exact Probability: {y_pred:.6f}")
print(f"Probability (rounded to 3 decimal places): {round(y_pred, 3)}")