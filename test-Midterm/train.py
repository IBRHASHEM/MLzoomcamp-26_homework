import pandas as pd
import xgboost as xgb
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

print("جارٍ قراءة البيانات...")
# 1. قراءة البيانات وتنظيفها
df = pd.read_csv('Customer-Churn-Records.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['rownumber', 'customerid', 'surname'])

# 2. تحديد المتغيرات
numerical = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'point_earned']
categorical = ['geography', 'gender', 'hascrcard', 'isactivemember', 'complain', 'satisfaction_score', 'card_type']

for col in categorical:
    df[col] = df[col].astype(str)

# 3. التقسيم (Train / Val) 
# في الإنتاج، عادة ندمج Train+Val معاً، لكن للتبسيط سنقوم بتدريبه كما فعلنا سابقاً
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y_train = df_full_train['exited'].values
del df_full_train['exited']

print("جارٍ تشفير البيانات وتدريب النموذج (XGBoost)...")
# 4. التشفير باستخدام DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# 5. تنظيف أسماء الأعمدة لتلائم XGBoost
feature_names = dv.get_feature_names_out().tolist()
clean_feature_names = [re.sub(r'[\[\]<>=]', '_', f) for f in feature_names]

# 6. بناء هيكل DMatrix وتدريب النموذج
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=clean_feature_names)

xgb_params = {
    'eta': 0.1, 
    'max_depth': 4, 
    'min_child_weight': 1, 
    'objective': 'binary:logistic', 
    'eval_metric': 'auc', 
    'nthread': 8, 
    'seed': 42
}

model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100)

print("الانتهاء من التدريب. جارٍ الحفظ...")
# 7. الحفظ في ملف ثنائي باستخدام pickle
output_file = 'churn_xgb_model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_xgb), f_out)

print(f"تم حفظ أداة التشفير والنموذج بنجاح في الملف: {output_file}")