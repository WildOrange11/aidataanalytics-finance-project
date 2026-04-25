import pandas as pd
import numpy as np
import joblib

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

clf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
profile_names = joblib.load("profile_names.pkl")

PROFILE_ADJUSTMENT = {
    "Low Risk":      0,
    "Moderate Risk": -5,
    "High Risk":     -10,
    "Critical Risk": -15,
}

RISK_PENALTY = 5


def engineer(data):
    df = pd.DataFrame([data])
    df["total_income"] = df["monthly_income"] + df["financial_aid"]
    df["discretionary_expenses"] = df[discretionary_cols].sum(axis=1)
    df["aid_dependency"] = (df["financial_aid"] / df["total_income"].replace(0, np.nan)).clip(0, 1)
    df["tuition_aid_coverage"] = (df["financial_aid"] / df["tuition"].replace(0, np.nan)).clip(0, 1)
    df["disc_to_income"] = (df["discretionary_expenses"] / df["monthly_income"].replace(0, np.nan)).clip(0, 5)
    df["income_stress"] = (df["discretionary_expenses"] / df["total_income"].replace(0, np.nan)).clip(0, 3)
    df["housing_stress"] = (df["housing"] / df["monthly_income"].replace(0, np.nan)).clip(0, 3)
    for col in discretionary_cols:
        df[col + "_ratio"] = (df[col] / df["monthly_income"].replace(0, np.nan)).clip(0, 2)
    return df


def compute_formula_score(monthly_income, discretionary_expenses, financial_aid):
    total_income = monthly_income + financial_aid
    disc_ratio = (discretionary_expenses / monthly_income) if monthly_income > 0 else 2
    aid_dependency = (financial_aid / total_income) if total_income > 0 else 1
    expense_norm = max(0, 1 - disc_ratio / 2)
    savings_norm = monthly_income / (monthly_income + discretionary_expenses) if (monthly_income + discretionary_expenses) > 0 else 0
    aid_penalty = 1 - aid_dependency * 0.3
    return float(np.clip((0.5 * expense_norm + 0.35 * savings_norm + 0.15 * aid_penalty) * 100, 0, 100))


def predict(data):
    df = engineer(data)

    X_num = scaler.transform(df[numeric_fit_cols])
    X_num = pd.DataFrame(X_num, columns=numeric_fit_cols)
    X_enc = pd.get_dummies(df[categorical_cols], drop_first=True)
    X_final = pd.concat([X_num, X_enc], axis=1)

    for col in feature_columns:
        if col not in X_final.columns:
            X_final[col] = 0
    X_final = X_final[feature_columns]

    cluster_id = clf.predict(X_final)[0]
    confidence = round(float(clf.predict_proba(X_final)[0].max()) * 100, 1)
    profile = profile_names[cluster_id]

    disc_expenses = float(df["discretionary_expenses"].values[0])
    formula_score = compute_formula_score(data["monthly_income"], disc_expenses, data["financial_aid"])

    cluster_adj = PROFILE_ADJUSTMENT[profile]
    score = float(np.clip(formula_score + cluster_adj - RISK_PENALTY, 0, 100))
    score = min(score, formula_score)
    score = round(score, 2)

    if score >= 58:
        rating = "Healthy"
    elif score >= 32:
        rating = "Moderate"
    else:
        rating = "At Risk"

    return {
        "spending_profile": profile,
        "confidence": confidence,
        "financial_score": score,
        "rating": rating
    }


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
    print("\n--- Student Financial Profiler ---\n")

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
    print(f"Spending profile : {result['spending_profile']} ({result['confidence']}% confidence)")
    print(f"Financial score  : {result['financial_score']} / 100")
    print(f"Rating           : {result['rating']}")