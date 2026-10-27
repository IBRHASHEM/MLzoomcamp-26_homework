import pickle
import xgboost as xgb # أو استورد ما تحتاجه للوجستي
from fastapi import FastAPI
from pydantic import BaseModel
import re

# 1. تهيئة تطبيق FastAPI
app = FastAPI(title="Churn Prediction API")

# 2. تحميل النموذج
model_file = 'churn_xgb_model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# 3. إعداد الـ Schema المتوقعة للعميل (بناءً على الأعمدة الخاصة بك)
# كمثال، استخدمت بيانات العملاء التي أرسلتها مسبقاً
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

# 4. نقطة نهاية أساسية للتأكد من حالة السيرفر
@app.get("/health")
def health():
    return {"status": "ok"}

# 5. نقطة التنبؤ الرئيسية
@app.post("/predict")
def predict(customer: Customer):
    # تحويل كائن العميل إلى قاموس
    customer_dict = customer.model_dump()
    
    # تحويل البيانات الفئوية لنصوص (إذا لم تكن كذلك)
    categorical = ['geography', 'gender', 'hascrcard', 'isactivemember', 'complain', 'satisfaction_score', 'card_type']
    for col in categorical:
        customer_dict[col] = str(customer_dict.get(col, 'NA'))
        
    # تشفير العميل
    X = dv.transform([customer_dict])
    
    # تنظيف الأعمدة إذا كنت تستخدم XGBoost
    feature_names = dv.get_feature_names_out().tolist()
    clean_feature_names = [re.sub(r'[\[\]<>=]', '_', f) for f in feature_names]
    dmatrix = xgb.DMatrix(X, feature_names=clean_feature_names)
    
    # التنبؤ بالاحتمالية
    prob = float(model.predict(dmatrix)[0])
    
    # تجهيز رد JSON
    return {
        "churn_probability": prob,
        "will_churn": bool(prob >= 0.5)
    }