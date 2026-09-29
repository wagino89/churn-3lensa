"""Create MOCK data with the same column schema as the real datasets.

Only for smoke tests: checks that the whole pipeline runs before the real
datasets are downloaded. Numbers obtained on mock data are MEANINGLESS and
must not be reported.

    python tests/make_mock_data.py --out data/mock
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def _label(logit):
    return rng.random(len(logit)) < 1 / (1 + np.exp(-logit))


def telco(n=7043):
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    monthly = rng.uniform(18, 119, n).round(2)
    total = (monthly * np.maximum(tenure, 0)).round(2).astype(str)
    total[tenure == 0] = " "
    logit = -1.2 + 1.3 * (contract == "Month-to-month") - 0.04 * tenure + 0.012 * monthly
    yes_no = lambda: rng.choice(["Yes", "No"], n)
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-MOCK" for i in range(n)], "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n), "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure,
        "PhoneService": yes_no(), "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": yes_no(), "OnlineBackup": yes_no(), "DeviceProtection": yes_no(), "TechSupport": yes_no(),
        "StreamingTV": yes_no(), "StreamingMovies": yes_no(), "Contract": contract, "PaperlessBilling": yes_no(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total,
    })
    df["Churn"] = np.where(_label(logit), "Yes", "No")
    return df


def bank(n=10000):
    age = rng.integers(18, 92, n)
    active = rng.integers(0, 2, n)
    nprod = rng.choice([1, 2, 3, 4], n, p=[0.5, 0.46, 0.03, 0.01])
    logit = -3.2 + 0.06 * age - 0.9 * active + 1.2 * (nprod >= 3)
    return pd.DataFrame({
        "RowNumber": np.arange(1, n + 1), "CustomerId": rng.integers(15_000_000, 16_000_000, n),
        "Surname": rng.choice(["Hargrave", "Hill", "Onio", "Boni", "Mitchell"], n),
        "CreditScore": rng.integers(350, 851, n), "Geography": rng.choice(["France", "Spain", "Germany"], n),
        "Gender": rng.choice(["Male", "Female"], n), "Age": age, "Tenure": rng.integers(0, 11, n),
        "Balance": np.round(rng.choice([0, 1], n) * rng.uniform(0, 250000, n), 2), "NumOfProducts": nprod,
        "HasCrCard": rng.integers(0, 2, n), "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(10, 200000, n).round(2), "Exited": _label(logit).astype(int)})


def bankchurners(n=10127):
    trans_ct = rng.integers(10, 140, n)
    logit = 1.5 - 0.045 * trans_ct + rng.normal(0, 0.5, n)
    y = _label(logit)
    df = pd.DataFrame({
        "CLIENTNUM": rng.integers(7e8, 8e8, n),
        "Attrition_Flag": np.where(y, "Attrited Customer", "Existing Customer"),
        "Customer_Age": rng.integers(26, 74, n), "Gender": rng.choice(["M", "F"], n),
        "Dependent_count": rng.integers(0, 6, n),
        "Education_Level": rng.choice(["Graduate", "High School", "Unknown", "Uneducated"], n),
        "Marital_Status": rng.choice(["Married", "Single", "Unknown"], n),
        "Income_Category": rng.choice(["Less than $40K", "$40K - $60K", "$80K - $120K", "Unknown"], n),
        "Card_Category": rng.choice(["Blue", "Silver", "Gold"], n, p=[0.93, 0.05, 0.02]),
        "Months_on_book": rng.integers(13, 57, n), "Total_Relationship_Count": rng.integers(1, 7, n),
        "Months_Inactive_12_mon": rng.integers(0, 7, n), "Contacts_Count_12_mon": rng.integers(0, 7, n),
        "Credit_Limit": rng.uniform(1400, 35000, n).round(1), "Total_Revolving_Bal": rng.integers(0, 2600, n),
        "Total_Trans_Amt": (trans_ct * rng.uniform(30, 90, n)).round(0), "Total_Trans_Ct": trans_ct,
        "Total_Ct_Chng_Q4_Q1": rng.uniform(0, 1.5, n).round(3), "Avg_Utilization_Ratio": rng.uniform(0, 1, n).round(3),
    })
    # leaking columns as in the real dataset (must be removed by the loader)
    df["Naive_Bayes_Classifier_Attrition_Flag_1"] = np.where(y, 0.99, 0.01)
    df["Naive_Bayes_Classifier_Attrition_Flag_2"] = 1 - df["Naive_Bayes_Classifier_Attrition_Flag_1"]
    return df


def iranian(n=3150):
    complains = rng.integers(0, 2, n)
    use = rng.integers(0, 17000, n)
    logit = -2.3 + 2.2 * complains - 0.00015 * use
    return pd.DataFrame({
        "Call  Failure": rng.integers(0, 37, n), "Complains": complains,
        "Subscription  Length": rng.integers(3, 48, n), "Charge  Amount": rng.integers(0, 11, n),
        "Seconds of Use": use, "Frequency of use": rng.integers(0, 256, n),
        "Frequency of SMS": rng.integers(0, 523, n), "Distinct Called Numbers": rng.integers(0, 98, n),
        "Age Group": rng.integers(1, 6, n), "Tariff Plan": rng.integers(1, 3, n), "Status": rng.integers(1, 3, n),
        "Age": rng.choice([15, 25, 30, 45, 55], n), "Customer Value": rng.uniform(0, 2165, n).round(3),
        "Churn": _label(logit).astype(int)})


def bigml(n=3333):
    cs_calls = rng.integers(0, 10, n)
    intl = rng.choice(["yes", "no"], n, p=[0.1, 0.9])
    logit = -2.8 + 0.5 * cs_calls + 1.5 * (intl == "yes")
    df = pd.DataFrame({
        "state": rng.choice(["KS", "OH", "NJ", "OK", "AL"], n), "account length": rng.integers(1, 244, n),
        "area code": rng.choice([408, 415, 510], n),
        "phone number": [f"{rng.integers(300, 999)}-{rng.integers(1000, 9999)}" for _ in range(n)],
        "international plan": intl, "voice mail plan": rng.choice(["yes", "no"], n),
        "number vmail messages": rng.integers(0, 51, n), "total day minutes": rng.uniform(0, 350, n).round(1),
        "total day calls": rng.integers(0, 165, n), "total eve minutes": rng.uniform(0, 363, n).round(1),
        "total night minutes": rng.uniform(23, 395, n).round(1), "total intl minutes": rng.uniform(0, 20, n).round(1),
        "customer service calls": cs_calls})
    df["churn"] = _label(logit)
    return df


def retail(n_customers=1500):
    """Mock transaction log 2009-12-01 to 2011-12-09 with hazard-based churn behaviour."""
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    rows = []
    for cid in range(12346, 12346 + n_customers):
        t = start + pd.Timedelta(days=int(rng.integers(0, 500)))
        rate = rng.uniform(5, 60)       # mean days between purchases
        hazard = rng.uniform(0.02, 0.2)  # probability of stopping after each purchase
        country = "United Kingdom" if rng.random() < 0.85 else rng.choice(["Germany", "France", "EIRE"])
        inv = 0
        while t < end:
            inv += 1
            for _ in range(int(rng.integers(1, 6))):
                rows.append((f"{cid}{inv:04d}", f"S{rng.integers(100, 400)}", int(rng.integers(1, 24)),
                             t + pd.Timedelta(minutes=int(rng.integers(0, 600))),
                             round(float(rng.uniform(0.5, 15)), 2), cid, country))
            if rng.random() < 0.05:
                rows.append((f"C{cid}{inv:04d}", "S100", -1, t + pd.Timedelta(days=1), 2.5, cid, country))
            if rng.random() < hazard:
                break
            t += pd.Timedelta(days=int(rng.exponential(rate)) + 1)
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price",
                                       "Customer ID", "Country"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mock")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    telco().to_csv(out / "WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)
    bank().to_csv(out / "Churn_Modelling.csv", index=False)
    bankchurners().to_csv(out / "BankChurners.csv", index=False)
    iranian().to_csv(out / "Customer Churn.csv", index=False)
    bigml().to_csv(out / "bigml_mock.csv", index=False)
    retail().to_csv(out / "online_retail_II_mock.csv", index=False)
    print(f"Mock data written to {out}/ (do NOT report as results)")


if __name__ == "__main__":
    main()
