import pandas as pd
import xgboost as xgb
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

print("Reading dataset...")
# 1. Read and clean dataset
df = pd.read_csv('Customer-Churn-Records.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['rownumber', 'customerid', 'surname'])

# 2. Define features
numerical = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'point_earned']
categorical = ['geography', 'gender', 'hascrcard', 'isactivemember', 'complain', 'satisfaction_score', 'card_type']

for col in categorical:
    df[col] = df[col].astype(str)

# 3. Train/Test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y_train = df_full_train['exited'].values
del df_full_train['exited']

print("Encoding features and training XGBoost model...")
# 4. Feature encoding using DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# 5. Clean feature names for XGBoost compatibility
feature_names = dv.get_feature_names_out().tolist()
clean_feature_names = [re.sub(r'[\[\]<>=]', '_', f) for f in feature_names]

# 6. Build DMatrix and train model
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

print("Training finished. Saving model...")
# 7. Save DictVectorizer and model to binary pickle file
output_file = 'churn_xgb_model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_xgb), f_out)

print(f"Model and DictVectorizer successfully saved to: {output_file}")