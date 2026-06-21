import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model_pipeline.pkl"


class CreditScoreModel:
    def __init__(self, pkl_path: str = MODEL_PATH):
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


@st.cache_resource
def load_model():
    return CreditScoreModel(MODEL_PATH)


st.set_page_config(page_title="Credit Score Classification", page_icon="💳", layout="centered")

st.title("💳 Credit Score Classification")
st.caption("Prediksi kategori credit score secara langsung di dalam aplikasi Streamlit (tanpa backend API terpisah).")

try:
    model_wrapper = load_model()
    st.success(f"Model '{model_wrapper.model_name}' berhasil dimuat dari {MODEL_PATH}")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.form("credit_score_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Annual_Income = st.number_input("Annual Income", min_value=0.0, value=45000.0)
        Monthly_Inhand_Salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=3500.0)
        Num_Bank_Accounts = st.number_input("Num Bank Accounts", min_value=0, value=4)
        Num_Credit_Card = st.number_input("Num Credit Card", min_value=0, value=3)
        Interest_Rate = st.number_input("Interest Rate", min_value=0, value=12)
        Num_of_Loan = st.number_input("Num of Loan", min_value=0, value=2)
        Delay_from_due_date = st.number_input("Delay from due date", value=10)
        Num_of_Delayed_Payment = st.number_input("Num of Delayed Payment", min_value=0, value=3)

    with col2:
        Changed_Credit_Limit = st.number_input("Changed Credit Limit", value=5.5)
        Num_Credit_Inquiries = st.number_input("Num Credit Inquiries", min_value=0, value=2)
        Outstanding_Debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0)
        Credit_Utilization_Ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, value=30.5)
        Credit_History_Age = st.number_input("Credit History Age (bulan)", min_value=0, value=120)
        Total_EMI_per_month = st.number_input("Total EMI per month", min_value=0.0, value=150.0)
        Amount_invested_monthly = st.number_input("Amount invested monthly", min_value=0.0, value=200.0)
        Monthly_Balance = st.number_input("Monthly Balance", value=400.0)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        Occupation = st.selectbox(
            "Occupation",
            ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer",
             "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant",
             "Musician", "Mechanic", "Writer", "Architect"],
            index=2,
        )
        Credit_Mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])

    with col4:
        Payment_of_Min_Amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
        Payment_Behaviour = st.selectbox(
            "Payment Behaviour",
            ["High_spent_Small_value_payments", "High_spent_Medium_value_payments",
             "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
             "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments"],
        )

    submitted = st.form_submit_button("Prediksi")

if submitted:
    data = {
        "Age": Age,
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": Num_Bank_Accounts,
        "Num_Credit_Card": Num_Credit_Card,
        "Interest_Rate": Interest_Rate,
        "Num_of_Loan": Num_of_Loan,
        "Delay_from_due_date": Delay_from_due_date,
        "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": Num_Credit_Inquiries,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Credit_History_Age": Credit_History_Age,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Monthly_Balance": Monthly_Balance,
        "Occupation": Occupation,
        "Credit_Mix": Credit_Mix,
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Payment_Behaviour": Payment_Behaviour,
    }

    try:
        result = model_wrapper.predict(data)
        st.subheader("Hasil Prediksi")
        st.metric("Credit Score", result["prediction"])

        if result.get("probabilities"):
            proba_df = pd.DataFrame(
                {
                    "Kelas": list(result["probabilities"].keys()),
                    "Probabilitas": list(result["probabilities"].values()),
                }
            ).sort_values("Probabilitas", ascending=False)
            st.bar_chart(proba_df.set_index("Kelas"))
            st.dataframe(proba_df, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
