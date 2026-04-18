import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("student_spending (1).csv")

expense_cols = ["tuition", "housing", "food", "transportation",
                "books_supplies", "entertainment", "personal_care",
                "technology", "health_wellness", "miscellaneous"]

df["total_expenses"] = df[expense_cols].sum(axis=1)
df["total_income"] = df["monthly_income"] + df["financial_aid"]
df["savings"] = df["monthly_income"] - df["total_expenses"]

df["expense_ratio"] = (df["total_expenses"] / df["total_income"].replace(0, np.nan)).clip(0, 2)
df["savings_rate"] = (df["savings"] / df["monthly_income"].replace(0, np.nan)).clip(-3, 1)
df["aid_dependency"] = df["financial_aid"] / df["total_income"].replace(0, np.nan)

expense_norm = 1 - df["expense_ratio"].clip(0, 1)
savings_norm = (df["savings_rate"].clip(-1, 1) + 1) / 2
aid_penalty = 1 - df["aid_dependency"].clip(0, 1) * 0.3

df["financial_score"] = (
    0.5 * expense_norm +
    0.35 * savings_norm +
    0.15 * aid_penalty
) * 100

for col in expense_cols:
    df[col + "_ratio"] = (df[col] / df["monthly_income"].replace(0, np.nan)).clip(0, 2)

df["housing_burden"] = df["housing_ratio"] > 0.3
df["food_burden"] = df["food_ratio"] > 0.15
df["high_aid_reliance"] = df["aid_dependency"] > 0.4

categorical_cols = ["gender", "year_in_school", "major", "preferred_payment_method"]

target = "financial_score"

leak_cols = ["savings", "savings_rate", "expense_ratio", "total_expenses", "total_income"]

X = df.drop(columns=expense_cols + [target] + leak_cols)
Y = df[target]

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = RobustScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(X.shape)
print(Y.shape)
print(X.head())

processed_df = X.copy()
processed_df["financial_score"] = Y.values

processed_df.to_csv("student_spending--PROCESSED.csv", index=False)