import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class CreditScoreModel:
    

    def __init__(self, pkl_path: str = "model_pipeline.pkl"):
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {pkl_path}. "
                "Jalankan pipeline.py terlebih dahulu untuk membuat model_pipeline.pkl"
            )

        with open(pkl_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.num_imputer = artifact["num_imputer"]
        self.cat_imputer = artifact["cat_imputer"]
        self.le_dict = artifact["le_dict"]
        self.scaler = artifact["scaler"]
        self.label_encoder = artifact["label_encoder"]
        self.numeric_features = artifact["numeric_features"]
        self.categorical_features = artifact["categorical_features"]
        self.all_features = artifact["all_features"]
        self.classes = artifact["classes"]

    def _preprocess(self, record: dict) -> pd.DataFrame:
        row = pd.DataFrame([record])

        for col in self.all_features:
            if col not in row.columns:
                row[col] = np.nan
        X = row[self.all_features].copy()

        X[self.numeric_features] = self.num_imputer.transform(X[self.numeric_features])
        X[self.categorical_features] = self.cat_imputer.transform(X[self.categorical_features])

        for col in self.categorical_features:
            le = self.le_dict[col]
            X[col] = X[col].astype(str).map(lambda v, le=le: v if v in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X

    def predict(self, record: dict) -> dict:
        
        X = self._preprocess(record)

        pred_enc = self.model.predict(X)[0]
        pred_label = self.label_encoder.inverse_transform([pred_enc])[0]

        result = {"prediction": str(pred_label)}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            result["probabilities"] = {
                str(cls): float(p)
                for cls, p in zip(self.label_encoder.classes_, proba)
            }

        return result


if __name__ == "__main__":
    model = CreditScoreModel("model_pipeline.pkl")
    sample = {
        "Age": 35, "Annual_Income": 45000.0, "Monthly_Inhand_Salary": 3500.0,
        "Num_Bank_Accounts": 4, "Num_Credit_Card": 3, "Interest_Rate": 12,
        "Num_of_Loan": 2, "Delay_from_due_date": 10, "Num_of_Delayed_Payment": 3,
        "Changed_Credit_Limit": 5.5, "Num_Credit_Inquiries": 2,
        "Outstanding_Debt": 1200.0, "Credit_Utilization_Ratio": 30.5,
        "Credit_History_Age": 120, "Total_EMI_per_month": 150.0,
        "Amount_invested_monthly": 200.0, "Monthly_Balance": 400.0,
        "Occupation": "Engineer", "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "High_spent_Small_value_payments",
    }
    print(model.predict(sample))
