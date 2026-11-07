
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("../healthcare-dataset-stroke-data_manipulated.csv")
MODEL_PATH = Path("stroke_prediction_model.bin")

TARGET = "stroke"
ID_COLUMN = "id"

NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

# Normalize column names.
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# Normalize categorical text.
raw_categorical = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

for column in raw_categorical:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

# Validate required columns.
required_columns = [ID_COLUMN, TARGET] + FEATURES
missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Train / validation / test split
# --------------------------------------------------

df_train_full, df_test = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TARGET],
)

df_train, df_val = train_test_split(
    df_train_full,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=df_train_full[TARGET],
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Training rows:   {len(df_train)}")
print(f"Validation rows: {len(df_val)}")
print(f"Test rows:       {len(df_test)}")


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

# Learn imputation values from training data only.
numerical_medians = df_train[NUMERICAL_FEATURES].median()
categorical_modes = df_train[CATEGORICAL_FEATURES].mode(
    dropna=True
).iloc[0]


def preprocess_features(frame):
    result = frame[FEATURES].copy()

    for column in NUMERICAL_FEATURES:
        result[column] = result[column].fillna(
            numerical_medians[column]
        )

    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna(
            categorical_modes[column]
        )
        result[column] = result[column].astype(str)

    # Fixed BMI upper cap used in the notebook.
    result["bmi"] = result["bmi"].clip(upper=60)

    return result


X_train_df = preprocess_features(df_train)
X_val_df = preprocess_features(df_val)
X_test_df = preprocess_features(df_test)

y_train = df_train[TARGET].astype(int).to_numpy()
y_val = df_val[TARGET].astype(int).to_numpy()
y_test = df_test[TARGET].astype(int).to_numpy()


# --------------------------------------------------
# 5. Encode features
# --------------------------------------------------

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(
    X_train_df.to_dict(orient="records")
)

X_val = dv.transform(
    X_val_df.to_dict(orient="records")
)

X_test = dv.transform(
    X_test_df.to_dict(orient="records")
)

feature_names = dv.get_feature_names_out()

print(f"Encoded features: {len(feature_names)}")


# --------------------------------------------------
# 6. Logistic Regression
# --------------------------------------------------

baseline_lr = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=RANDOM_STATE,
)

baseline_lr.fit(X_train, y_train)

lr_val_prob = baseline_lr.predict_proba(X_val)[:, 1]
lr_val_auc = roc_auc_score(y_val, lr_val_prob)

print(f"Logistic Regression AUC: {lr_val_auc:.4f}")

# Preserve the notebook's Logistic Regression tuning experiment.
lr_tuning_results = []

for c_value in [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100
]:
    candidate = LogisticRegression(
        solver="liblinear",
        C=c_value,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    candidate.fit(X_train, y_train)

    probabilities = candidate.predict_proba(X_val)[:, 1]

    lr_tuning_results.append({
        "C": c_value,
        "validation_auc": roc_auc_score(y_val, probabilities),
    })


# --------------------------------------------------
# 7. Decision Tree tuning
# --------------------------------------------------

dt_results = []

for depth in [3, 5, 7, 10, None]:
    for leaf in [1, 5, 10, 20, 50]:

        candidate = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )

        candidate.fit(X_train, y_train)

        probabilities = candidate.predict_proba(X_val)[:, 1]

        dt_results.append({
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "validation_auc": roc_auc_score(y_val, probabilities),
        })

dt_results_df = pd.DataFrame(dt_results).sort_values(
    "validation_auc",
    ascending=False,
).reset_index(drop=True)

best_dt_params = dt_results_df.iloc[0]

best_dt = DecisionTreeClassifier(
    max_depth=(
        None
        if pd.isna(best_dt_params["max_depth"])
        else int(best_dt_params["max_depth"])
    ),
    min_samples_leaf=int(best_dt_params["min_samples_leaf"]),
    random_state=RANDOM_STATE,
)

best_dt.fit(X_train, y_train)

dt_val_prob = best_dt.predict_proba(X_val)[:, 1]
dt_val_auc = roc_auc_score(y_val, dt_val_prob)

print(f"Decision Tree AUC: {dt_val_auc:.4f}")


# --------------------------------------------------
# 8. Random Forest tuning
# --------------------------------------------------

rf_results = []

for n_estimators in [100, 200, 300]:
    for depth in [5, 7, 10, None]:
        for leaf in [5, 10, 20]:

            candidate = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            candidate.fit(X_train, y_train)

            probabilities = candidate.predict_proba(X_val)[:, 1]

            rf_results.append({
                "n_estimators": n_estimators,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "validation_auc": roc_auc_score(y_val, probabilities),
            })

rf_results_df = pd.DataFrame(rf_results).sort_values(
    "validation_auc",
    ascending=False,
).reset_index(drop=True)

rf_best = rf_results_df.iloc[0]

best_rf = RandomForestClassifier(
    n_estimators=int(rf_best["n_estimators"]),
    max_depth=(
        None
        if pd.isna(rf_best["max_depth"])
        else int(rf_best["max_depth"])
    ),
    min_samples_leaf=int(rf_best["min_samples_leaf"]),
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

best_rf.fit(X_train, y_train)

rf_val_prob = best_rf.predict_proba(X_val)[:, 1]
rf_val_auc = roc_auc_score(y_val, rf_val_prob)

print(f"Random Forest AUC: {rf_val_auc:.4f}")


# --------------------------------------------------
# 9. XGBoost tuning
# --------------------------------------------------

xgb_results = []

for n_estimators in [100, 200, 300]:
    for depth in [2, 3, 4, 5]:
        for learning_rate in [0.01, 0.05, 0.1]:

            candidate = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                learning_rate=learning_rate,
                objective="binary:logistic",
                eval_metric="logloss",
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            candidate.fit(X_train, y_train)

            probabilities = candidate.predict_proba(X_val)[:, 1]

            xgb_results.append({
                "n_estimators": n_estimators,
                "max_depth": depth,
                "learning_rate": learning_rate,
                "validation_auc": roc_auc_score(y_val, probabilities),
            })

xgb_results_df = pd.DataFrame(xgb_results).sort_values(
    "validation_auc",
    ascending=False,
).reset_index(drop=True)

xgb_best = xgb_results_df.iloc[0]

best_xgb = XGBClassifier(
    n_estimators=int(xgb_best["n_estimators"]),
    max_depth=int(xgb_best["max_depth"]),
    learning_rate=float(xgb_best["learning_rate"]),
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

best_xgb.fit(X_train, y_train)

xgb_val_prob = best_xgb.predict_proba(X_val)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_prob)

print(f"XGBoost AUC: {xgb_val_auc:.4f}")


# --------------------------------------------------
# 10. Compare models
# --------------------------------------------------

models = {
    "Logistic Regression": baseline_lr,
    "Decision Tree": best_dt,
    "Random Forest": best_rf,
    "XGBoost": best_xgb,
}

validation_probabilities = {
    name: model.predict_proba(X_val)[:, 1]
    for name, model in models.items()
}

comparison = pd.DataFrame([
    {
        "model": name,
        "validation_roc_auc": roc_auc_score(y_val, probabilities),
    }
    for name, probabilities in validation_probabilities.items()
]).sort_values(
    "validation_roc_auc",
    ascending=False,
).reset_index(drop=True)

print("\nModel comparison:")
print(comparison.to_string(index=False))

selected_model_name = comparison.iloc[0]["model"]
selected_model = models[selected_model_name]
selected_val_prob = validation_probabilities[selected_model_name]


# --------------------------------------------------
# 11. Select decision threshold using validation data
# --------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

threshold_rows = []

for threshold in np.arange(0.10, 0.91, 0.05):
    predictions = (selected_val_prob >= threshold).astype(int)

    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    })

threshold_results = pd.DataFrame(threshold_rows)

threshold_results = threshold_results.sort_values(
    ["f1", "threshold"],
    ascending=[False, True],
).reset_index(drop=True)

best_threshold = float(threshold_results.iloc[0]["threshold"])

print(f"\nSelected model: {selected_model_name}")
print(f"Selected threshold: {best_threshold:.2f}")


# --------------------------------------------------
# 12. Final evaluation on test data
# --------------------------------------------------

test_probabilities = selected_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = {
    "model": selected_model_name,
    "threshold": best_threshold,
    "roc_auc": roc_auc_score(y_test, test_probabilities),
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
}

print("\nTest metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")


# --------------------------------------------------
# 13. Save model package
# --------------------------------------------------

model_package = {
    "model": selected_model,
    "dv": dv,
    "threshold": best_threshold,
    "model_name": selected_model_name,
    "numerical_medians": numerical_medians.to_dict(),
    "categorical_modes": categorical_modes.to_dict(),
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "feature_columns": FEATURES,
    "bmi_upper_cap": 60,
    "random_state": RANDOM_STATE,
    "test_metrics": test_metrics,
}

with open(MODEL_PATH, "wb") as f_out:
    pickle.dump(model_package, f_out)

print(f"\nModel package saved: {MODEL_PATH.resolve()}")
print(f"Package size: {MODEL_PATH.stat().st_size:,} bytes")

