import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

expense_cols = ["tuition", "housing", "food", "transportation",
                "books_supplies", "entertainment", "personal_care",
                "technology", "health_wellness", "miscellaneous"]

discretionary_cols = ["housing", "food", "transportation", "books_supplies",
                      "entertainment", "personal_care", "technology",
                      "health_wellness", "miscellaneous"]

categorical_cols = ["gender", "year_in_school", "major", "preferred_payment_method"]

numeric_fit_cols = ["age", "monthly_income", "financial_aid",
                    "aid_dependency", "tuition_aid_coverage",
                    "housing_ratio", "food_ratio", "transportation_ratio",
                    "books_supplies_ratio", "entertainment_ratio",
                    "personal_care_ratio", "technology_ratio",
                    "health_wellness_ratio", "miscellaneous_ratio",
                    "housing_burden", "food_burden", "high_aid_reliance"]


def engineer(df):
    df = df.copy()
    df["total_income"] = df["monthly_income"] + df["financial_aid"]
    df["discretionary_expenses"] = df[discretionary_cols].sum(axis=1)
    df["aid_dependency"] = (df["financial_aid"] / df["total_income"].replace(0, np.nan)).clip(0, 1)
    df["tuition_aid_coverage"] = (df["financial_aid"] / df["tuition"].replace(0, np.nan)).clip(0, 1)
    for col in discretionary_cols:
        df[col + "_ratio"] = (df[col] / df["monthly_income"].replace(0, np.nan)).clip(0, 2)
    df["housing_burden"] = (df["housing_ratio"] > 0.3).astype(int)
    df["food_burden"] = (df["food_ratio"] > 0.15).astype(int)
    df["high_aid_reliance"] = (df["aid_dependency"] > 0.4).astype(int)
    return df


df = pd.read_csv("student_spending (1).csv")
df = engineer(df)

disc_ratio = (df["discretionary_expenses"] / df["monthly_income"].replace(0, np.nan))
disc_ratio_clipped = disc_ratio.clip(lower=0)

p_low = disc_ratio_clipped.quantile(0.05)
p_high = disc_ratio_clipped.quantile(0.95)

expense_norm = 1 - ((disc_ratio_clipped - p_low) / (p_high - p_low)).clip(0, 1)
savings_norm = (df["monthly_income"] / (df["monthly_income"] + df["discretionary_expenses"].replace(0, np.nan))).clip(0, 1)
aid_penalty = 1 - df["aid_dependency"] * 0.3

raw_score = (0.5 * expense_norm + 0.35 * savings_norm + 0.15 * aid_penalty)

s_min = raw_score.min()
s_max = raw_score.max()
df["financial_score"] = ((raw_score - s_min) / (s_max - s_min) * 100).round(2)

print("--- Rescaled financial_score distribution ---")
print(df["financial_score"].describe().round(2))
print(f"p25: {df['financial_score'].quantile(0.25):.2f}")
print(f"p50: {df['financial_score'].quantile(0.50):.2f}")
print(f"p75: {df['financial_score'].quantile(0.75):.2f}")

drop_cols = expense_cols + ["discretionary_expenses", "total_income", "financial_score"]
X = df.drop(columns=drop_cols)
Y = df["financial_score"]

scaler = RobustScaler()
X[numeric_fit_cols] = scaler.fit_transform(X[numeric_fit_cols])
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
)
model.fit(X_train, Y_train)

from sklearn.metrics import mean_absolute_error, r2_score
Y_pred = model.predict(X_test)
print(f"\nMAE: {mean_absolute_error(Y_test, Y_pred):.4f}")
print(f"R2:  {r2_score(Y_test, Y_pred):.4f}")

joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
joblib.dump({"s_min": s_min, "s_max": s_max, "p_low": p_low, "p_high": p_high}, "score_params.pkl")
print("\nModel saved.")