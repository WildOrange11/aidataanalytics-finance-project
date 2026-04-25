import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
                    "disc_to_income", "housing_ratio", "food_ratio",
                    "transportation_ratio", "books_supplies_ratio",
                    "entertainment_ratio", "personal_care_ratio",
                    "technology_ratio", "health_wellness_ratio",
                    "miscellaneous_ratio", "income_stress"]

df = pd.read_csv("student_spending (1).csv")

df["total_income"] = df["monthly_income"] + df["financial_aid"]
df["discretionary_expenses"] = df[discretionary_cols].sum(axis=1)
df["aid_dependency"] = (df["financial_aid"] / df["total_income"].replace(0, np.nan)).clip(0, 1)
df["tuition_aid_coverage"] = (df["financial_aid"] / df["tuition"].replace(0, np.nan)).clip(0, 1)
df["disc_to_income"] = (df["discretionary_expenses"] / df["monthly_income"].replace(0, np.nan)).clip(0, 5)
df["income_stress"] = (df["discretionary_expenses"] / df["total_income"].replace(0, np.nan)).clip(0, 3)
df["housing_stress"] = (df["housing"] / df["monthly_income"].replace(0, np.nan)).clip(0, 3)

for col in discretionary_cols:
    df[col + "_ratio"] = (df[col] / df["monthly_income"].replace(0, np.nan)).clip(0, 2)

cluster_features = [
    "disc_to_income",
    "income_stress",
    "housing_stress",
    "aid_dependency",
    "housing_ratio",
    "food_ratio",
    "entertainment_ratio",
    "tuition_aid_coverage",
]

cluster_weights = np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

cluster_scaler = RobustScaler()
X_cluster_raw = cluster_scaler.fit_transform(df[cluster_features])
X_cluster = X_cluster_raw * cluster_weights

kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
df["spending_profile"] = kmeans.fit_predict(X_cluster)

profile_means = df.groupby("spending_profile")[
    ["disc_to_income", "income_stress", "aid_dependency", "monthly_income", "housing_stress"]
].mean().round(3)
print("--- Cluster profiles ---")
print(profile_means)

stress_order = profile_means["income_stress"].sort_values()
profile_names = {}
labels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
for rank, (cluster_id, _) in enumerate(stress_order.items()):
    profile_names[cluster_id] = labels[rank]

df["profile_name"] = df["spending_profile"].map(profile_names)
print("\n--- Profile distribution ---")
print(df["profile_name"].value_counts())

X = df[numeric_fit_cols].copy()
Y = df["spending_profile"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_fit_cols)

X_enc = pd.get_dummies(df[categorical_cols], drop_first=True)
X_final = pd.concat([X_scaled, X_enc], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("\n--- Classifier report ---")
print(classification_report(Y_test, Y_pred, target_names=[profile_names[i] for i in sorted(profile_names)]))

importances = pd.Series(clf.feature_importances_, index=X_final.columns)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

joblib.dump(clf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_final.columns), "feature_columns.pkl")
joblib.dump(cluster_scaler, "cluster_scaler.pkl")
joblib.dump(kmeans, "kmeans.pkl")
joblib.dump(profile_names, "profile_names.pkl")
joblib.dump(cluster_weights, "cluster_weights.pkl")
print("\nAll models saved.")