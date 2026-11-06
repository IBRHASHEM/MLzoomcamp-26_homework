
import pickle
from flask import Flask, request, jsonify

# Load the trained model package
with open('stroke_prediction_model.bin', 'rb') as f_in:
    package = pickle.load(f_in)

model = package['model']
dv = package['dv']
threshold = package['threshold']

# Initialize Flask application
app = Flask('stroke-prediction')


@app.route('/predict', methods=['POST'])
def predict():

    # Receive patient data
    patient = request.get_json()

    # Transform input data
    X = dv.transform([patient])

    # Predict stroke probability
    probability = float(model.predict_proba(X)[0, 1])

    # Apply decision threshold
    prediction = int(probability >= threshold)

    # Return prediction result
    result = {
        'stroke_probability': round(probability, 4),
        'stroke_prediction': prediction,
        'threshold': threshold
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=False)
