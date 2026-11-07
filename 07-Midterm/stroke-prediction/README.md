# Stroke Prediction — Machine Learning Midterm Project

**Machine Learning Zoomcamp 2026**

## 1. Project Overview

This project develops an end-to-end machine learning classification system to estimate the likelihood of stroke based on patient demographic information and health indicators.

The project demonstrates a complete machine learning workflow, including:

- Data exploration and analysis (EDA).
- Data cleaning and preprocessing.
- Feature engineering and encoding.
- Training multiple classification models.
- Hyperparameter tuning.
- Model comparison and selection.
- Decision threshold optimization.
- Final evaluation on a held-out test dataset.
- Model serialization.
- REST API development using Flask.
- Containerization and deployment using Docker.

**Disclaimer:** This project is for educational purposes only. It is not intended for medical diagnosis, clinical screening, treatment decisions, or emergency decision-making.

## 2. Dataset

The project uses a manipulated version of the Stroke Prediction dataset.

**Dataset information:**

- Total records: 4,909
- Total columns: 12
- Target variable: `stroke`

The dataset contains the following features:

| Feature | Description |
|---|---|
| gender | Patient gender |
| age | Patient age |
| hypertension | Hypertension indicator |
| heart_disease | Heart disease indicator |
| ever_married | Marital status |
| work_type | Type of employment |
| Residence_type | Urban or rural residence |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking category |
| id | Record identifier, excluded from model predictors |
| stroke | Target variable |

**Dataset filename:**

`healthcare-dataset-stroke-data_manipulated.csv`

The dataset is included in the `07-Midterm` directory and a copy is also available inside the `stroke-prediction` project directory.

**Dataset paths:**

- Training script: reads the dataset from the parent `07-Midterm` directory.
- Notebook: the notebook instructions specify placing the dataset in the same directory as the notebook.

## 3. Machine Learning Workflow

The project follows a structured and reproducible machine learning workflow.

1. Load and inspect the dataset.
2. Perform initial data quality checks.
3. Explore the target variable and predictor distributions.
4. Perform exploratory data analysis (EDA).
5. Split the dataset into training, validation, and test sets.
6. Handle missing values using training-set statistics.
7. Apply BMI cleaning using a fixed upper cap of 60.
8. Encode categorical and numerical features using `DictVectorizer`.
9. Train multiple classification models.
10. Tune model hyperparameters using validation data.
11. Compare candidate models using validation ROC AUC.
12. Select the best-performing model.
13. Optimize the classification threshold using validation F1 score.
14. Evaluate the selected model on the held-out test set.
15. Inspect feature importance.
16. Save and reload the trained model package.
17. Integrate the model with a Flask prediction API.
18. Build and run the application using Docker.

### Data Splitting

The dataset is divided using stratified splitting:

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Test | 20% |

The test dataset is reserved for final evaluation and is not used for model selection or threshold tuning.

## 4. Models Evaluated

Four classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Hyperparameter tuning was performed for Decision Tree, Random Forest, and XGBoost. Logistic Regression was used as the baseline model.

### Validation ROC AUC Comparison

| Model | Validation ROC AUC |
|---|---:|
| XGBoost | 0.9838 |
| Random Forest | 0.9831 |
| Decision Tree | 0.9779 |
| Logistic Regression | 0.9715 |

**Selected model:** XGBoost

**Selected decision threshold:** 0.45

The model was selected based on validation ROC AUC. The classification threshold was selected by maximizing validation F1 score.

## 5. Final Model Performance

The selected XGBoost model was evaluated on the held-out test dataset.

### Test Metrics

| Metric | Test Result |
|---|---:|
| ROC AUC | 0.9898 |
| Accuracy | 0.9807 |
| Precision | 0.9657 |
| Recall | 0.9426 |
| F1 Score | 0.9540 |
| Decision Threshold | 0.45 |

**Selected model:** XGBoost

These metrics represent the results obtained from this project's specific test split. They should not be interpreted as evidence of clinical effectiveness or generalization to other populations.

## 6. Project Screenshots

The following screenshots demonstrate the project workflow and deployment.

### Exploratory Data Analysis

![EDA Analysis](screenshots/01-eda-analysis.png)

### API Prediction

![API Prediction](screenshots/02-api-prediction.png)

### Docker Deployment

![Docker Deployment](screenshots/03-docker-container.png)

## 7. Project Structure

```text
MLzoomcamp-26_homework/
│
└── 07-Midterm/
    │
    ├── healthcare-dataset-stroke-data_manipulated.csv
    │
    └── stroke-prediction/
        │
        ├── notebook.ipynb
        ├── train.py
        ├── predict.py
        ├── stroke_prediction_model.bin
        ├── healthcare-dataset-stroke-data_manipulated.csv
        ├── requirements.txt
        ├── Dockerfile
        ├── README.md
        │
        └── screenshots/
            ├── 01-eda-analysis.png
            ├── 02-api-prediction.png
            └── 03-docker-container.png
```

### Main Files

| File | Purpose |
|---|---|
| `notebook.ipynb` | Exploratory analysis, model experiments, evaluation, and documentation |
| `train.py` | Reproducible model training and serialization |
| `predict.py` | Flask REST API for model predictions |
| `stroke_prediction_model.bin` | Serialized model package |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker image configuration |
| `screenshots/` | Project screenshots |

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Flask
- Gunicorn
- Docker
- GitHub Codespaces
- Jupyter Notebook

## 9. Environment Setup

### Prerequisites

- Python 3.11 or a compatible Python version.
- pip.
- Docker (for containerized deployment).
- GitHub Codespaces or a local development environment.

### Install Dependencies

From the `stroke-prediction` directory, run:

```bash
pip install -r requirements.txt
```

## 10. Training the Model

The training script reads the dataset from the parent `07-Midterm` directory.

From the `stroke-prediction` directory, execute:

```bash
python train.py
```

The script performs the following operations:

1. Loads the dataset.
2. Splits the data into training, validation, and test sets.
3. Performs preprocessing and feature encoding.
4. Trains and compares the classification models.
5. Selects the best model using validation ROC AUC.
6. Selects the decision threshold using validation F1 score.
7. Evaluates the selected model on the test dataset.
8. Saves the trained model package.

The resulting model package is saved as:

```text
stroke_prediction_model.bin
```

## 11. Running the API with Docker

The Flask application is served using Gunicorn inside a Docker container.

### Build the Docker Image

From the `stroke-prediction` directory:

```bash
docker build -t stroke-prediction:latest .
```

### Run the Container

```bash
docker run -d \
  --name stroke-prediction-api \
  -p 9696:9696 \
  stroke-prediction:latest
```

### Check Container Status

```bash
docker ps
```

### Check Application Logs

```bash
docker logs stroke-prediction-api
```

### Stop the Container

```bash
docker stop stroke-prediction-api
```

### Remove the Container

```bash
docker rm stroke-prediction-api
```

If the container name is already in use, stop and remove the existing container before creating another one.

## 12. API Usage

The application exposes a REST API endpoint for stroke prediction.

**HTTP Method:** `POST`

**Endpoint:**

```text
/predict
```

**Local URL:**

```text
http://localhost:9696/predict
```

### Example JSON Request

```json
{
  "gender": "female",
  "age": 67,
  "hypertension": 0,
  "heart_disease": 0,
  "ever_married": "yes",
  "work_type": "private",
  "residence_type": "urban",
  "avg_glucose_level": 120.5,
  "bmi": 28.4,
  "smoking_status": "never smoked"
}
```

### Example API Response

```json
{
  "model": "XGBoost",
  "stroke_prediction": 0,
  "stroke_probability": 0.0969,
  "threshold": 0.45
}
```

### Test the API Using cURL

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "urban",
    "avg_glucose_level": 120.5,
    "bmi": 28.4,
    "smoking_status": "never smoked"
  }'
```

### API Response Interpretation

| Field | Description |
|---|---|
| model | Selected machine learning model |
| stroke_prediction | Predicted class (0 or 1) |
| stroke_probability | Estimated probability of the positive class |
| threshold | Decision threshold used for classification |

The API was successfully tested locally and inside the Docker container.

## 13. Reproducibility

To reproduce the project:

1. Clone or download the project repository.
2. Ensure the dataset is available in the expected location.
3. Install dependencies using `requirements.txt`.
4. Run `train.py` from the `stroke-prediction` directory.
5. Build the Docker image.
6. Run the Docker container.
7. Send a POST request to the `/predict` endpoint.

The training script uses a fixed random seed to support reproducible data splitting and experiments.

## 14. Limitations and Future Improvements

### Current Limitations

- The dataset represents a specific collection of records and may not represent all populations.
- High evaluation scores do not guarantee real-world performance.
- The model may be affected by dataset bias, missing information, and class imbalance.
- Feature importance does not establish causal relationships.
- The selected threshold is based on validation F1 score, not clinical requirements.
- The API is an educational demonstration and is not designed or validated for clinical use.

### Possible Future Improvements

- Further investigate dataset quality and potential data leakage.
- Evaluate model calibration and generalization on independent datasets.
- Perform more detailed fairness and subgroup analysis.
- Improve API validation, logging, and error handling.
- Deploy the application to a cloud platform.

## 15. Author

**Ibrahim Hashem**

Machine Learning Zoomcamp 2026 — Midterm Project

---

**Medical and Ethical Disclaimer**

This project is intended exclusively for educational and learning purposes. The predictions generated by this model are not medical advice and must not be used for diagnosis, screening, treatment, or emergency decisions. Always consult qualified healthcare professionals for medical evaluation.