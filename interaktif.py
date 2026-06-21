import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Credit Score Classification",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.header-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    margin-bottom: 20px;
}

.metric-card {
    background-color: white;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0px 1px 10px rgba(0,0,0,0.08);
}

.risk-good{
    background:#dcfce7;
    padding:20px;
    border-radius:10px;
}

.risk-standard{
    background:#fef9c3;
    padding:20px;
    border-radius:10px;
}

.risk-poor{
    background:#fee2e2;
    padding:20px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>💳 Credit Score Classification</h1>
    <p>
        Prediksi kategori Credit Score menggunakan Machine Learning Model
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:

    st.title("⚙️ System")

    st.markdown("---")

    st.subheader("Backend Status")

    try:
        health = requests.get(
            f"{API_URL}/health",
            timeout=3
        ).json()

        if health.get("model_loaded"):
            st.success("Backend Connected ✅")
            st.info(
                f"Model: {health.get('model_name')}"
            )
        else:
            st.warning("Model not loaded")
    except:
        st.error("Backend Offline ❌")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        Sistem ini memprediksi:
        - Good
        - Standard
        - Poor

        berdasarkan profil kredit nasabah.
        """
    )


tab1, tab2 = st.tabs(
    [
        "📊 Financial Information",
        "🏦 Credit Information"
    ]
)

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            100,
            35
        )

        annual_income = st.number_input(
            "Annual Income",
            value=45000.0
        )

        monthly_salary = st.number_input(
            "Monthly Salary",
            value=3500.0
        )

        amount_invested = st.number_input(
            "Monthly Investment",
            value=200.0
        )

        monthly_balance = st.number_input(
            "Monthly Balance",
            value=400.0
        )

    with col2:

        occupation = st.selectbox(
            "Occupation",
            [
                "Engineer",
                "Doctor",
                "Teacher",
                "Lawyer",
                "Scientist",
                "Accountant",
                "Manager",
                "Entrepreneur",
                "Journalist",
                "Mechanic",
                "Developer",
                "Media_Manager",
                "Architect",
                "Writer",
                "Musician",
                "Lainnya"
            ]
        )

        outstanding_debt = st.number_input(
            "Outstanding Debt",
            value=1200.0
        )

        total_emi = st.number_input(
            "Total EMI",
            value=150.0
        )

with tab2:

    col3, col4 = st.columns(2)

    with col3:

        num_bank_accounts = st.number_input(
            "Bank Accounts",
            value=4
        )

        num_credit_card = st.number_input(
            "Credit Cards",
            value=3
        )

        num_of_loan = st.number_input(
            "Loans",
            value=2
        )

        interest_rate = st.number_input(
            "Interest Rate",
            value=12.0
        )

        delay_from_due_date = st.number_input(
            "Delay From Due Date",
            value=10
        )

    with col4:

        num_delayed_payment = st.number_input(
            "Delayed Payments",
            value=3
        )

        changed_credit_limit = st.number_input(
            "Changed Credit Limit",
            value=5.5
        )

        num_credit_inquiries = st.number_input(
            "Credit Inquiries",
            value=2
        )

        credit_util_ratio = st.number_input(
            "Credit Utilization Ratio",
            value=30.5
        )

        credit_history_months = st.number_input(
            "Credit History (Months)",
            value=120
        )

credit_mix = st.selectbox(
    "Credit Mix",
    ["Good", "Standard", "Bad"]
)

payment_min_amount = st.selectbox(
    "Payment of Minimum Amount",
    ["Yes", "No"]
)

payment_behaviour = st.selectbox(
    "Payment Behaviour",
    [
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments"
    ]
)


st.subheader("📈 Applicant Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Age",
    age
)

c2.metric(
    "Income",
    f"${annual_income:,.0f}"
)

c3.metric(
    "Debt",
    f"${outstanding_debt:,.0f}"
)

c4.metric(
    "Loans",
    num_of_loan
)


health_score = 100

health_score -= min(
    outstanding_debt / 1000 * 5,
    30
)

health_score -= min(
    delay_from_due_date,
    20
)

health_score -= min(
    num_delayed_payment * 2,
    20
)

health_score = max(
    0,
    round(health_score)
)

st.metric(
    "💡 Credit Health Score",
    f"{health_score}/100"
)


payload = {
    "Age": age,
    "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary,
    "Num_Bank_Accounts": num_bank_accounts,
    "Num_Credit_Card": num_credit_card,
    "Interest_Rate": interest_rate,
    "Num_of_Loan": num_of_loan,
    "Delay_from_due_date": delay_from_due_date,
    "Num_of_Delayed_Payment": num_delayed_payment,
    "Changed_Credit_Limit": changed_credit_limit,
    "Num_Credit_Inquiries": num_credit_inquiries,
    "Outstanding_Debt": outstanding_debt,
    "Credit_Utilization_Ratio": credit_util_ratio,
    "Credit_History_Age": credit_history_months,
    "Total_EMI_per_month": total_emi,
    "Amount_invested_monthly": amount_invested,
    "Monthly_Balance": monthly_balance,
    "Occupation": occupation,
    "Credit_Mix": credit_mix,
    "Payment_of_Min_Amount": payment_min_amount,
    "Payment_Behaviour": payment_behaviour,
}

with st.expander("📄 Review Submitted Data"):
    st.json(payload)


if st.button(
    "🚀 Predict Credit Score",
    use_container_width=True
):

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=20
        )

        if response.status_code == 200:

            result = response.json()
            pred = result["prediction"]

            st.divider()

            st.header("🎯 Prediction Result")

            score_map = {
                "Poor": 25,
                "Standard": 65,
                "Good": 95
            }

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score_map[pred],
                    title={'text': "Credit Score"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'steps': [
                            {'range':[0,40]},
                            {'range':[40,70]},
                            {'range':[70,100]}
                        ]
                    }
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            if pred == "Good":

                st.markdown("""
                <div class="risk-good">
                <h3>🟢 Good Credit Score</h3>
                Risiko kredit rendah dan profil keuangan sehat.
                </div>
                """,
                unsafe_allow_html=True)

            elif pred == "Standard":

                st.markdown("""
                <div class="risk-standard">
                <h3>🟡 Standard Credit Score</h3>
                Masih terdapat beberapa risiko yang perlu diperhatikan.
                </div>
                """,
                unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="risk-poor">
                <h3>🔴 Poor Credit Score</h3>
                Risiko kredit tinggi.
                </div>
                """,
                unsafe_allow_html=True)

            if result.get("probabilities"):

                prob_df = pd.DataFrame(
                    result["probabilities"].items(),
                    columns=[
                        "Class",
                        "Probability"
                    ]
                )

                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    text="Probability",
                    title="Prediction Confidence"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("📌 Recommendations")

            if pred == "Poor":

                st.error("""
                • Kurangi outstanding debt

                • Hindari keterlambatan pembayaran

                • Kurangi jumlah pinjaman aktif

                • Tingkatkan saldo bulanan
                """)

            elif pred == "Standard":

                st.warning("""
                • Pertahankan pembayaran tepat waktu

                • Kurangi rasio penggunaan kredit

                • Tingkatkan investasi bulanan
                """)

            else:

                st.success("""
                • Pertahankan performa kredit

                • Jaga rasio kredit tetap sehat

                • Pertimbangkan investasi tambahan
                """)

            with st.expander("🔍 Full API Response"):
                st.json(result)

        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Error: {e}")

