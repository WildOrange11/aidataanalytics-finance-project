import pandas as pd
import numpy as np
import joblib

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

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


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


def preprocess_input(data):
    df = pd.DataFrame([data])
    df = engineer(df)
    drop_cols = ["tuition", "housing", "food", "transportation", "books_supplies",
                 "entertainment", "personal_care", "technology", "health_wellness",
                 "miscellaneous", "discretionary_expenses", "total_income"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df[numeric_fit_cols] = scaler.transform(df[numeric_fit_cols])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


def predict(data):
    X = preprocess_input(data)
    score = round(float(np.clip(model.predict(X)[0], 0, 100)), 2)
    if score >= 60:
        rating = "Healthy"
    elif score >= 30:
        rating = "Moderate"
    else:
        rating = "At Risk"
    return {"financial_score": score, "rating": rating}


def get_input(prompt, cast):
    while True:
        try:
            return cast(input(prompt).strip())
        except ValueError:
            print("  Invalid input, try again.")


def get_choice(prompt, options):
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        try:
            idx = int(input("Enter number: ").strip())
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


if __name__ == "__main__":
    print("\nStudent Financial Score Predictor\n")

    data = {}
    data["age"] = get_input("Age: ", int)
    data["monthly_income"] = get_input("Monthly income ($): ", float)
    data["financial_aid"] = get_input("Financial aid ($): ", float)
    data["tuition"] = get_input("Tuition ($): ", float)

    print("\n-- Monthly Expenses --")
    data["housing"] = get_input("Housing ($): ", float)
    data["food"] = get_input("Food ($): ", float)
    data["transportation"] = get_input("Transportation ($): ", float)
    data["books_supplies"] = get_input("Books & supplies ($): ", float)
    data["entertainment"] = get_input("Entertainment ($): ", float)
    data["personal_care"] = get_input("Personal care ($): ", float)
    data["technology"] = get_input("Technology ($): ", float)
    data["health_wellness"] = get_input("Health & wellness ($): ", float)
    data["miscellaneous"] = get_input("Miscellaneous ($): ", float)

    print()
    data["gender"] = get_choice("Gender:", ["Male", "Female", "Non-binary"])
    data["year_in_school"] = get_choice("Year in school:", ["Freshman", "Sophomore", "Junior", "Senior"])
    data["major"] = get_choice("Major:", ["Biology", "Computer Science", "Economics", "Engineering", "Psychology"])
    data["preferred_payment_method"] = get_choice("Preferred payment method:", ["Cash", "Credit/Debit Card", "Mobile Payment App"])

    result = predict(data)
    print(f"\n--- Result ---")
    print(f"Financial Score : {result['financial_score']} / 100")
    print(f"Rating          : {result['rating']}")