# Churn Prediction — Trading Platform Users

## Problem
The retention team wants to identify users likely to churn (stop trading) in the next
30 days, so they can prioritize proactive outreach. This project builds a model that
scores each user's churn probability and deploys it as a simple interactive demo the
retention team can use directly.

**Live Streamlit APP** : https://churn-prediction-app-streamlit.onrender.com

## Data
- Source: `data/sample_churn.csv` — 3,000 users, target column `churn`
- Churn rate: **12%** (imbalanced — a "predict everyone stays" model would score 88%
  accuracy while being useless, which is why accuracy isn't the metric we lead with)
- Columns dropped: `user_id` — a unique identifier with no predictive value
- Missing values: `account_balance`, `avg_trade_size`, `num_support_tickets` (~3% each).
  Checked whether missingness itself was informative (churn rate for missing vs.
  non-missing rows) — differences were within noise given the small sample, so
  straightforward median (numeric) / mode (categorical) imputation was used.
- Leakage check: reviewed every column for whether it would actually be known at
  prediction time (before the user churns). No leakage found — all remaining features
  are observable in real time.

## Feature engineering
Three features were added, all row-level arithmetic (safe to compute before the
train/test split, since nothing is learned across rows):
- `net_flow_30d` = `deposit_30d` − `withdrawal_30d` — net cash flow signal
- `trade_intensity` = `num_trades_30d` / `tenure_days` — trading activity relative to
  account age
- `tickets_per_100_days` = `num_support_tickets` / `tenure_days` × 100 — support
  friction normalized by tenure

## Approach
1. **Split first** (80/20, stratified on `churn`), then fit all imputation/scaling/
   encoding statistics on the training set only, to avoid test-set leakage into
   preprocessing.
2. Preprocessing (median/mode imputation, scaling, one-hot encoding) is embedded
   inside an sklearn `Pipeline`, so training and inference always apply identical
   transforms — no train/serve skew.
3. **Baseline**: Logistic Regression, `class_weight="balanced"` (to counter the 12%
   minority class).
4. **Improved candidate**: Random Forest, same class balancing, depth-limited to
   reduce overfitting risk on ~2,400 training rows.
5. Model selection by **ROC-AUC** (threshold-independent, robust to class imbalance),
   cross-checked against recall given the business cost of missed churners.

## Why these metrics, not accuracy
With a 12% churn rate, accuracy is misleading — a model that never predicts churn
still scores ~88%. Instead:
- **False negative cost** (missing an actual churner): lost revenue, no chance to
  intervene — the costlier error for this use case.
- **False positive cost** (flagging a user who wouldn't have churned): one wasted
  retention touchpoint — comparatively cheap.
- Given that asymmetry, **recall is prioritized** over precision, with ROC-AUC used
  as the primary model-comparison metric.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.703 | 0.255 | **0.764** | 0.382 | **0.801** |
| Random Forest (improved) | 0.810 | 0.319 | 0.514 | 0.394 | 0.775 |

**Selected model: Logistic Regression.** It wins on both the primary comparison
metric (ROC-AUC: 0.801 vs. 0.775) and the business-critical metric (recall: 76% of
churners caught vs. 51%). Random Forest likely underperformed here due to limited
training examples of the minority class (~290 churned users in training) — with more
data, the more flexible model would be expected to close or reverse this gap. This was
a genuine result, not a default choice: the more complex model was evaluated and
explicitly not selected, because it didn't earn its added complexity on this dataset.

Confusion matrix (test set, 600 users, 72 actual churners), Logistic Regression:

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actual stay** | 367 | 161 |
| **Actual churn** | 17 | 55 |

## Top churn drivers (Logistic Regression coefficients)

| Feature | Coefficient | Direction |
|---|---|---|
| `tenure_days` | −0.800 | Longer tenure → lower churn risk |
| `is_verified` | ∓0.44 | Unverified users churn at ~2× the rate of verified users |
| `net_flow_30d` | −0.350 | Net deposits (vs. withdrawals) → lower churn risk |
| `num_support_tickets` | +0.292 | More support contact → higher churn risk |
| `withdrawal_30d` | +0.282 | Higher withdrawals → higher churn risk |
| `deposit_30d` | −0.228 | Higher deposits → lower churn risk |
| `account_type` / `country` | small, mixed | Weak/inconsistent in EDA — treat with caution |

**Business interpretation:** the strongest churn signals are low tenure, lack of
verification, and 30-day cash-flow/support-contact behavior. Retention efforts should
prioritize newly-onboarded, unverified users showing net withdrawals or recent support
tickets — this group is disproportionately likely to leave in the next 30 days.

## Assumptions & limitations
- Missing values assumed non-informative based on the churn-rate-by-missingness check;
  revisit if the real dataset shows a starker gap.
- `country` and `account_type` showed weak or inconsistent effects in EDA — not
  dropped, but not treated as reliable drivers either.
- Trained on a single snapshot of data; would want to validate stability over time and
  monitor for feature drift before relying on this in production.
- Random Forest's underperformance may not generalize — worth revisiting with more
  training data or hyperparameter tuning rather than ruling it out permanently.

## Next steps
- Threshold tuning against actual business cost, rather than the default 0.5 cutoff
- SHAP values for per-prediction explanations to support individual retention actions
- Retrain cadence and drift monitoring once in production

## How to run
```bash
pip install -r requirements.txt
python train.py                  # trains, evaluates, saves saved_model.pkl + metrics.json
streamlit run app.py             # launches the demo UI
```

## Project structure
```
app.py              # Streamlit demo interface
model.py            # Model loading + prediction utilities
train.py            # Data loading, cleaning, feature engineering, training, evaluation
config.py           # Data path, target column, drop columns
requirements.txt
data/sample_churn.csv
saved_model.pkl      # Trained pipeline (generated by train.py)
metrics.json         # Model metrics + metadata (generated by train.py)
```
