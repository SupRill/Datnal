import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Hapus kolom target jika ada
    if "Churn" in df.columns:
        df_features = df.drop(columns=["Churn"])
    else:
        df_features = df.copy()

    if st.button("Predict"):
        preds = model.predict(df_features)
        
        # Jika model punya probabilitas
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_features)[:, 1]
            df["Prediction_Prob"] = prob
        
        df["Prediction"] = preds
        st.subheader("Prediction Result")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "result.csv", "text/csv")
