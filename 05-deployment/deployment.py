# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# #url="https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# %%
# df=pd.read_csv(url)
df = pd.read_csv('data-week-3.csv')
df.columns=df.columns.str.lower().str.replace(' ','_')
categorical_columns=list(df.dtypes[df.dtypes=='object'].index)
for c in categorical_columns:
    df[c]=df[c].str.lower().str.replace(' ','_')
df.totalcharges=pd.to_numeric(df.totalcharges,errors='coerce')
df.totalcharges=df.totalcharges.fillna(0)
df.churn=(df.churn=='yes').astype(int)

# %%
df_full_train, df_test=train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val=train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train=df_train.churn.values
y_val=df_val.churn.values
y_test=df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# %%
numerical=['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 1. إعادة قراءة ملف البيانات من جديد
df = pd.read_csv('data-week-3.csv')

# 2. تنظيف أسماء الأعمدة والقيم النصية
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.select_dtypes(include=['object', 'str']).columns)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# 3. معالجة totalcharges وتحويل churn إلى 0 و 1
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)
df['churn'] = (df['churn'] == 'yes').astype(int)

# 4. تحديد المتغيرات
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection',
    'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod'
]

# 5. تقسيم البيانات (60% Train, 20% Val, 20% Test)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# 6. استخراج الهدف قبل حذفه من الـ DataFrame
y_train = df_train['churn'].values
y_val = df_val['churn'].values
y_test = df_test['churn'].values

del df_train['churn']
del df_val['churn']
del df_test['churn']

# التحقق من وجود الفئتين 0 و 1
print("y_train distribution:\n", pd.Series(y_train).value_counts())

# 7. التشفير عبر DictVectorizer والتدريب
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

print("\nModel fitted successfully!")

# %%
dv=DictVectorizer(sparse=False)
train_dict=df_train[categorical + numerical].to_dict(orient='records')
X_train=dv.fit_transform(train_dict)

model=LogisticRegression()
model.fit(X_train, y_train)


# %%
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()

# %%
len(y_val)

# %%
(y_val==churn_decision).sum()

# %%
1131/1409

# %%
from sklearn.metrics import accuracy_score


# %%
accuracy_score(y_val, y_pred >= 0.5)

# %%
thresholds=np.linspace(0, 1, 21)
scores = []
for t in thresholds:
    score=accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)

# %%
plt.plot(thresholds, scores)

# %%
from collections import Counter
Counter(y_pred >= 1.0)

# %%
result = 1 - y_val.mean()
print(round(result, 5))


# %%
actual_positive =(y_val==1)
actual_negative=(y_val==0)



# %%
t=0.5
predict_positive=(y_pred>=t )
predict_negative=(y_pred<t )


# %%
predict_positive [:5]

# %%
actual_positive[:5]

# %%
tp=(predict_positive & actual_positive).sum()
tn=(predict_negative & actual_negative).sum()
fp=(predict_positive & actual_negative).sum()
fn=(predict_negative & actual_positive).sum()

# %%


# %%
cofustion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
cofustion_matrix

# %%
(cofustion_matrix / cofustion_matrix.sum()).round(2)

# %%
(tp +tn) /(tp+tn+fp+fn)

# %%
p = tp /(tp+fp)

# %%
p

# %%
tp

# %%
tp+fp

# %%
r=tp / (tp+fn)
r

# %%
tp+fn

# %%
tpr= tp /(tp+fn)
tpr

# %%
fpr=fp/(fp+tn)
fpr

# %%
scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))

# %%
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

# %%
tpr = tp / (tp + fn)
tpr
fpr = fp / (fp + tn)
fpr

# %%
df_scores[::10]

# %%
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()

# %%
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
((y_rand>=0.5)==y_val).mean()

# %%
def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores

# %%
df_rand=tpr-tpr_fpr_dataframe(y_val, y_rand)

# %%
df_rand

# %%
plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.legend()

# %%
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos

# %%
y_ideal= np.repeat([0,1], [num_neg, num_pos])

# %%
y_ideal_pred=np.linspace(0, 1, len(y_val))

# %%

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal

y_ideal_pred = np.linspace(0, 1, len(y_val))

# %%
1 - y_val.mean()

# %%
accuracy_score(y_ideal, y_ideal_pred >= 0.726)

df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]

# %%
plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.legend()

# %%
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')



plt.legend()

# %%
plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

# %%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# %%
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

# %%
from sklearn.metrics import auc
auc(fpr, tpr)

# %%
auc(df_scores.fpr, df_scores.tpr)

# %%
auc(df_ideal.fpr, df_ideal.tpr)

# %%
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)

# %%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred)

# %%
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

# %%
import random
n = 50000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n

# %%
n = 100000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()

# %%
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

# %%
dv, model = train(df_train, y_train, C=0.001)

# %%
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
C=1.0
n_splits=5

# %%
y_pred = predict(df_val, dv, model)

# %%
# !pip install tqdm


# %%
from sklearn.model_selection import KFold
from tqdm.auto import tqdm               

# %%
n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# %%
scores

# %%
import warnings
warnings.filterwarnings('ignore')

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

# %%
# save the model

# %%
import pickle

# %%
output_file = f'model_C={C}.bin'



# %%
output_file

# %%
f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()

# %%
#!ls -lh *.bin

# %%
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)

# %%
import pickle

# %%
input_file = 'model_C=10.bin'

# %%
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

# %%
dv, model

# %%
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# %%
#X=dv.transform({customer})
X=dv.transform([customer])

# %%
#y_pred=model.predict_proba(X)[0,1}
y_pred=model.predict_proba(X)[0,1]

# %%
print('input:', customer)
print('output:', y_pred)

# %%
import requests

# %%
url = 'http://localhost:9696/predict'

# %%
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'two_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# %%
response = requests.post(url, json=customer).json()

# %%
response

# %%
if response['churn']:
    print('sending email to', 'asdx-123d')

# %%



