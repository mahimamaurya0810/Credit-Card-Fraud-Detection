import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Machine Learning based application to detect fraudulent "
    "credit card transactions."
)

st.divider()

st.subheader("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction features",
    type=["csv"]
)

required_features = [
    "V1", "V2", "V3", "V4", "V5",
    "V7", "V9", "V10", "V11", "V12",
    "V14", "V16", "V17", "V18"
]

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(data.head())

    missing_features = [
        feature for feature in required_features
        if feature not in data.columns
    ]

    if missing_features:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_features)
        )

    else:

        model = joblib.load("model.joblib")

        X = data[required_features]

        predictions = model.predict(X)

        data["Prediction"] = predictions

        data["Result"] = np.where(
            predictions == 1,
            "🚨 Fraudulent Transaction",
            "✅ Normal Transaction"
        )

        st.subheader("🔍 Prediction Results")

        st.dataframe(data)

        fraud_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "✅ Normal Transactions",
                normal_count
            )

        with col2:
            st.metric(
                "🚨 Fraudulent Transactions",
                fraud_count
            )

        if fraud_count > 0:
            st.warning(
                f"⚠️ {fraud_count} fraudulent transaction(s) detected!"
            )
        else:
            st.success(
                "🎉 No fraudulent transactions detected."
            )