# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:35:54 2026

@author: JR
"""
"----------------------------------------------------------------------------------------"
import pandas as pd
df = pd.read_csv("C:/Users/JR/OneDrive/Learning/PORTFOLIO PROJECTS/Assessment prep/sample_churn.csv")
df.shape
df.head()
df.dtypes

"Identifying Target"
df["churn"].value_counts(normalize=True)

"Missing Value"
df.isna().sum().sort_values(ascending=False)

import matplotlib.pyplot as plt

for col in ["account_balance", "avg_trade_size", "num_support_tickets"]:
    df[col].plot(kind="box", title=col)
    plt.show()

"Is the missingness itself informative"
for col in ["account_balance", "avg_trade_size", "num_support_tickets"]:
    print(col, "\n", df.groupby(df[col].isna())["churn"].mean(), "\n")
"----------------------------------------------------------------------------------------"
"EDA"
"Numeric feature distributions split by churn"
import matplotlib.pyplot as plt

numeric_cols = ["tenure_days", "num_trades_30d", "avg_trade_size", "deposit_30d",
                 "withdrawal_30d", "account_balance", "num_support_tickets"]

for col in numeric_cols:
    df.boxplot(column=col, by="churn")
    plt.title(col)
    plt.suptitle("")
    plt.show()

"Categorical churn rates"
for col in ["account_type", "country", "is_verified"]:
    print(df.groupby(col)["churn"].mean(), "\n")
    
"Correlation among numeric features (multicollinearity check)"
df[numeric_cols + ["churn"]].corr()["churn"].sort_values(ascending=False)

"Feature engineering"  
df["net_flow_30d"] = df["deposit_30d"] - df["withdrawal_30d"]        # withdrawing more than depositing = disengagement signal
df["trade_intensity"] = df["num_trades_30d"] / (df["tenure_days"] + 1)  # trades per day active, +1 avoids divide-by-zero
df["tickets_per_100_days"] = df["num_support_tickets"] / (df["tenure_days"] + 1) * 100  
    
"----------------------------------------------------------------------------------------"    
"Train-Test Split"    
from sklearn.model_selection import train_test_split

X = df.drop(columns=["user_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = ["tenure_days", "num_trades_30d", "avg_trade_size", "deposit_30d",
                 "withdrawal_30d", "account_balance", "num_support_tickets",
                 "net_flow_30d", "trade_intensity", "tickets_per_100_days"]
categorical_cols = ["is_verified", "account_type", "country"]

"----------------------------------------------------------------------------------------"
"Imputation"
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

"----------------------------------------------------------------------------------------"
"Model Building"
"Logistic Regression"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

baseline.fit(X_train, y_train)

y_pred_base = baseline.predict(X_test)
y_proba_base = baseline.predict_proba(X_test)[:, 1]

"Random Forest"
from sklearn.ensemble import RandomForestClassifier

improved = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])

improved.fit(X_train, y_train)

y_pred_rf = improved.predict(X_test)
y_proba_rf = improved.predict_proba(X_test)[:, 1]

"Evaluation"
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

def evaluate(name, y_true, y_pred, y_proba):
    print(f"--- {name} ---")
    print("accuracy: ", round(accuracy_score(y_true, y_pred), 4))
    print("precision:", round(precision_score(y_true, y_pred), 4))
    print("recall:   ", round(recall_score(y_true, y_pred), 4))
    print("f1:       ", round(f1_score(y_true, y_pred), 4))
    print("roc_auc:  ", round(roc_auc_score(y_true, y_proba), 4))
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))
    print()

evaluate("Logistic Regression", y_test, y_pred_base, y_proba_base)
evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf)

"----------------------------------------------------------------------------------------"
"Logistic Regression's predictions"
import numpy as np

# Get feature names after one-hot encoding
cat_encoder = baseline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
feature_names = numeric_cols + cat_names

coefs = baseline.named_steps["clf"].coef_[0]
importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

for name, coef in importance[:10]:
    direction = "increases" if coef > 0 else "decreases"
    print(f"{name:30s} {coef:+.3f}  ({direction} churn risk)")
    
"Save-Model"
import joblib
import json

joblib.dump(baseline, "saved_model.pkl")

metadata = {
    "model_type": "logistic_regression",
    "target_col": "churn",
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "categorical_options": {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_cols},
    "numeric_ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in numeric_cols},
    "metrics": {
        "logistic_regression": {"accuracy": 0.7033, "precision": 0.2546, "recall": 0.7639, "f1": 0.3819, "roc_auc": 0.8012},
        "random_forest": {"accuracy": 0.81, "precision": 0.319, "recall": 0.5139, "f1": 0.3936, "roc_auc": 0.7753},
    },
    "selected_model": "logistic_regression",
}
with open("metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)