import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import re

# 1. Initialize FastAPI application
app = FastAPI(title="Churn Prediction API")

# 2. Load serialized DictVectorizer and trained XGBoost model
model_file = 'churn_xgb_model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# 3. Define customer input schema using Pydantic
class Customer(BaseModel):
    geography: str
    gender: str
    hascrcard: int
    isactivemember: int
    complain: int
    satisfaction_score: int
    card_type: str
    creditscore: float
    age: int
    tenure: int
    balance: float
    numofproducts: int
    estimatedsalary: float
    point_earned: int

# 4. Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# 5. Main prediction endpoint
@app.post("/predict")
def predict(customer: Customer):
    # Convert Pydantic customer object to dictionary
    customer_dict = customer.model_dump()
    
    # Cast categorical features to string as done during training
    categorical = ['geography', 'gender', 'hascrcard', 'isactivemember', 'complain', 'satisfaction_score', 'card_type']
    for col in categorical:
        customer_dict[col] = str(customer_dict.get(col, 'NA'))
        
    # Encode features using DictVectorizer
    X = dv.transform([customer_dict])
    
    # Clean feature names for XGBoost compatibility
    feature_names = dv.get_feature_names_out().tolist()
    clean_feature_names = [re.sub(r'[\[\]<>=]', '_', f) for f in feature_names]
    dmatrix = xgb.DMatrix(X, feature_names=clean_feature_names)
    
    # Predict churn probability
    prob = float(model.predict(dmatrix)[0])
    
    # Prepare JSON response
    return {
        "churn_probability": prob,
        "will_churn": bool(prob >= 0.5)
    }