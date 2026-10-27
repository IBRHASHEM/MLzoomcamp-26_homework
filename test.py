# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression

# # 1. تحميل وتجهيز البيانات
# url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
# df = pd.read_csv(url)

# categorical = list(df.select_dtypes(include=['object', 'str']).columns)
# numerical = [c for c in df.select_dtypes(include=['number']).columns if c != 'converted']

# df[categorical] = df[categorical].fillna('NA')
# df[numerical] = df[numerical].fillna(0.0)
# df['converted'] = df['converted'].astype(int)

# # 2. تقسيم البيانات واستخراج y_train و y_val
# df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
# df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# y_train = df_train['converted'].values
# y_val = df_val['converted'].values

# features = categorical + numerical

# # 3. تجهيز الميزات وتدريب النموذج
# dv = DictVectorizer(sparse=False)
# X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))
# X_val = dv.transform(df_val[features].to_dict(orient='records'))

# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# model.fit(X_train, y_train)

# # استخراج الاحتمالات لفئة الهدف
# y_pred_proba = model.predict_proba(X_val)[:, 1]

# # -------------------------------------------------------------
# # 4. حساب العتبة المثلى بناءً على التكلفة (Threshold Tuning)
# # -------------------------------------------------------------
# cost_fp = 15    # تكلفة الإنذار الكاذب
# cost_fn = 200   # تكلفة تفويت العميل
# cost_tp = 20    # تكلفة التدخل الناجح
# cost_tn = 0

# thresholds = np.linspace(0.0, 1.0, 101)
# costs = []

# for t in thresholds:
#     actual_pos = (y_val == 1)
#     pred_pos = (y_pred_proba >= t)
    
#     tp = (pred_pos & actual_pos).sum()
#     fp = (pred_pos & ~actual_pos).sum()
#     fn = (~pred_pos & actual_pos).sum()
#     tn = (~pred_pos & ~actual_pos).sum()
    
#     total_cost = (fp * cost_fp) + (fn * cost_fn) + (tp * cost_tp) + (tn * cost_tn)
#     costs.append(total_cost)

# costs = np.array(costs)

# # العتبة الأقل تكلفة
# best_idx = np.argmin(costs)
# best_threshold = thresholds[best_idx]
# min_cost = costs[best_idx]

# print(f"العتبة المثلى: {best_threshold:.2f}")
# print(f"أدنى تكلفة إجمالية: ${min_cost:,.2f}")

# # 5. رسم المنحنى
# plt.figure(figsize=(7, 4))
# plt.plot(thresholds, costs, color='darkred', label='Total Cost')
# plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimal ({best_threshold:.2f})')
# plt.xlabel('Threshold')
# plt.ylabel('Cost ($)')
# plt.title('Cost vs Decision Threshold')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 1. تجهيز بيانات تطابق مثال الرسائل المزعجة (Spam)
# 1 = رسالة مزعجة (الإيجابي)، 0 = رسالة عادية (السلبي)

# القيم الحقيقية في الواقع (20 مزعجة و 80 عادية)
y_true = np.array([1]*20 + [0]*80)

# توقعات النموذج (15 TP و 5 FN) + (5 FP و 75 TN)
y_pred = np.array([1]*15 + [0]*5 + [1]*5 + [0]*75)

# 2. حساب وعرض مصفوفة الارتباك
cm = confusion_matrix(y_true, y_pred)
print("مصفوفة الارتباك (Confusion Matrix):")
print(cm)

# تفكيك المصفوفة إلى مكوناتها الأربعة
TN, FP, FN, TP = cm.ravel()
print(f"\nالسلبي الصحيح (TN): {TN}")
print(f"الإيجابي الخاطئ (FP): {FP}")
print(f"السلبي الخاطئ (FN): {FN}")
print(f"الإيجابي الصحيح (TP): {TP}\n")

# 3. حساب واستخراج المقاييس
print(f"الدقة العامة (Accuracy): {accuracy_score(y_true, y_pred):.2f}")
print(f"الوضوح (Precision): {precision_score(y_true, y_pred):.2f}")
print(f"الاستدعاء (Recall): {recall_score(y_true, y_pred):.2f}")