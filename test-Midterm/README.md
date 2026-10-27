# 🏦 Bank Customer Churn Prediction Service

> **Machine Learning Zoomcamp — Midterm Project**  
> An end-to-end Machine Learning project to predict bank customer churn, deployed as a REST API using FastAPI and containerized with Docker.

---

## 📋 Table of Contents
1. [Problem Description](#-problem-description)
2. [Dataset Overview](#-dataset-overview)
3. [Project Structure](#-project-structure)
4. [Exploratory Data Analysis & Modeling](#-exploratory-data-analysis--modeling)
5. [Environment Setup](#-environment-setup)
6. [Training the Model](#-training-the-model)
7. [Running the Web Service (FastAPI)](#-running-the-web-service-fastapi)
8. [Testing the Service](#-testing-the-service)
9. [Docker Deployment](#-docker-deployment)

---

## 🎯 Problem Description

Customer churn occurs when bank clients stop doing business or close their accounts. In the financial sector, acquiring new customers costs significantly more than retaining existing ones. 

The objective of this project is to develop an end-to-end machine learning system that:
- Analyzes customer demographic, financial, and behavioral patterns.
- Identifies early warning signs of churn (e.g., customer complaints, activity levels, balance).
- Predicts the probability of churn for any individual customer in real-time.
- Enables the bank's retention team to intervene proactively with personalized offers before a customer leaves.

---

## 📊 Dataset Overview

The dataset used is the **Bank Customer Churn Records** dataset (available on [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn-prediction)), containing 10,000 records of bank customers with 18 initial columns.

### Features:
- **Identifier columns (dropped during preprocessing):** `RowNumber`, `CustomerId`, `Surname`
- **Numerical Features (7):**
  - `creditscore`: Customer credit score
  - `age`: Customer age
  - `tenure`: Number of years with the bank
  - `balance`: Account balance
  - `numofproducts`: Number of bank products used
  - `estimatedsalary`: Estimated annual salary
  - `point_earned`: Points earned from bank transactions/card usage
- **Categorical Features (7):**
  - `geography`: Customer country (`France`, `Spain`, `Germany`)
  - `gender`: Gender (`Male`, `Female`)
  - `hascrcard`: Whether customer possesses a credit card (`0`, `1`)
  - `isactivemember`: Whether customer is an active member (`0`, `1`)
  - `complain`: Whether customer lodged a complaint (`0`, `1`)
  - `satisfaction_score`: Customer satisfaction rating (1 to 5)
  - `card_type`: Card tier (`DIAMOND`, `PLATINUM`, `GOLD`, `SILVER`)
- **Target Variable:**
  - `exited`: `1` if customer left the bank (churned), `0` if customer remained.

---

## 📁 Project Structure

```text
├── Customer-Churn-Records.csv   # Dataset file
├── midterm_churn_project.ipynb  # Jupyter Notebook for EDA, feature analysis & experimentation
├── train.py                     # Script to train final model and save it to pickle file
├── predict.py                   # FastAPI service for real-time model inference
├── predict_test.py              # Test script sending sample requests to predict endpoint
├── churn_xgb_model.bin          # Exported DictVectorizer + XGBoost model binary
├── Dockerfile                   # Docker container configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🔬 Exploratory Data Analysis & Modeling

In the exploratory phase ([`midterm_churn_project.ipynb`](./midterm_churn_project.ipynb)):
1. **Data Cleaning:** Column names were normalized to lowercase with underscores (`str.lower().str.replace(' ', '_')`), and redundant ID columns were eliminated.
2. **Feature Analysis:** Checked distributions, missing values, and feature importance. Strong correlation was observed between `complain` and `exited`, as well as `age` and `balance`.
3. **Encoding:** Categorical features were encoded using `DictVectorizer(sparse=False)`.
4. **Model Comparison:**
   - **Logistic Regression:**
     - Validation Accuracy: **77.49%**
     - Validation ROC AUC: **0.6772**
   - **XGBoost Classifier:**
     - Validation ROC AUC: **~0.999**
     - Test ROC AUC: **0.9978**

### Final Selected Model:
**XGBoost** with hyperparameters:
- `objective`: `binary:logistic`
- `eval_metric`: `auc`
- `eta`: `0.1`
- `max_depth`: `4`
- `min_child_weight`: `1`
- `num_boost_round`: `100`

The trained `DictVectorizer` and `model_xgb` are serialized together into `churn_xgb_model.bin`.

---

## ⚙️ Environment Setup

### 1. Clone the repository and navigate to the directory:
```bash
git clone <your-repo-url>
cd test-Midterm
```

### 2. Create and activate a virtual environment:
- **On Linux / macOS:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

To train the final model on the training data and export the serialized artifacts:

```bash
python train.py
```

**Output:**
```text
Reading dataset...
Encoding features and training XGBoost model...
Training finished. Saving model...
Model and DictVectorizer successfully saved to: churn_xgb_model.bin
```

---

## 🌐 Running the Web Service (FastAPI)

The inference API is built with **FastAPI** in [`predict.py`](./predict.py). Start the web server locally using **Uvicorn**:

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

Once running:
- **API URL:** `http://localhost:9696`
- **Health check:** `http://localhost:9696/health`
- **Interactive Swagger Documentation:** `http://localhost:9696/docs`

---

## 🧪 Testing the Service

With the server running in one terminal, open another terminal and run the test script:

```bash
python predict_test.py
```

### Example Request (`POST /predict`):
```json
{
  "geography": "France",
  "gender": "Female",
  "hascrcard": 1,
  "isactivemember": 1,
  "complain": 0,
  "satisfaction_score": 3,
  "card_type": "DIAMOND",
  "creditscore": 650.0,
  "age": 42,
  "tenure": 3,
  "balance": 75000.0,
  "numofproducts": 1,
  "estimatedsalary": 85000.0,
  "point_earned": 450
}
```

### Example Response:
```json
{
  "churn_probability": 0.0034,
  "will_churn": false
}
```

---

## 🐳 Docker Deployment

The application includes a [`Dockerfile`](./Dockerfile) based on `python:3.11-slim` for containerization.

### 1. Build the Docker image:
```bash
docker build -t churn-service .
```

### 2. Run the Docker container:
```bash
docker run -it --rm -p 9696:9696 churn-service
```

### 3. Test the containerized service:
Run `python predict_test.py` or use `curl`:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "geography": "Germany",
    "gender": "Female",
    "hascrcard": 1,
    "isactivemember": 0,
    "complain": 1,
    "satisfaction_score": 1,
    "card_type": "GOLD",
    "creditscore": 600.0,
    "age": 45,
    "tenure": 2,
    "balance": 120000.0,
    "numofproducts": 1,
    "estimatedsalary": 70000.0,
    "point_earned": 300
  }'
```

---

## 📝 Evaluation Criteria Checklist (ML Zoomcamp)

- [x] **Problem description:** Defined in detail with business motivation.
- [x] **EDA:** Performed in Jupyter notebook with feature importance and data cleaning.
- [x] **Multiple models evaluated:** Logistic Regression and XGBoost compared.
- [x] **Hyperparameter tuning:** Tuned for the final XGBoost model.
- [x] **Reproducibility:** `train.py` exports the model binary and vectorizer.
- [x] **Model deployment:** Implemented via FastAPI (`predict.py`).
- [x] **Dependency management:** Provided via `requirements.txt`.
- [x] **Containerization:** Complete `Dockerfile` ready to build and run.
