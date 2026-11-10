// Replace this URL if the Flask API is moved to another host.
const API_URL = "https://glorious-space-waddle-4jr7r5q4jw5rhqjrq-9697.app.github.dev/predict";

const form = document.getElementById("prediction-form");
const submitButton = document.getElementById("submit-button");
const message = document.getElementById("message");
const resultCard = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const patient = Object.fromEntries(formData.entries());

  for (const field of ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]) {
    patient[field] = Number(patient[field]);
  }

  message.hidden = true;
  resultCard.hidden = true;
  submitButton.disabled = true;
  submitButton.textContent = "Sending prediction...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `API request failed (HTTP ${response.status}).`);
    }

    document.getElementById("probability").textContent =
      `${(data.stroke_probability * 100).toFixed(2)}%`;
    document.getElementById("result-label").textContent =
      data.stroke_prediction === 1 ? "Positive Prediction" : "Negative Prediction";
    document.getElementById("result-detail").textContent =
      `Model: ${data.model} | Decision Threshold: ${data.threshold}`;

    resultCard.className = `result-card ${data.stroke_prediction === 1 ? "high-risk" : "low-risk"}`;
    resultCard.hidden = false;
  } catch (error) {
    message.textContent = `Unable to get a prediction: ${error.message}. Please check that the API service is running.`;
    message.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict Stroke Risk";
  }
});
