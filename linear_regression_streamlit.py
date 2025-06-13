import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "linear_reg_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    cols = joblib.load(os.path.join(base, "linear_reg_model_columns.pkl"))
    return model, scaler, cols

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "Food_Delivery_Times.csv"))
    df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)
    return df

model, scaler, cols = load_model()
df = load_data()

def preprocess_input(input_df, reference_columns):
    processed_df = input_df.copy()

    # Tambahkan kolom kosong untuk categorical encoded jika tidak dipakai
    for col in reference_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    processed_df = processed_df[reference_columns]
    return processed_df

def main():
    st.set_page_config(page_title="Simple Delivery Time Predictor", page_icon="⏱️")
    st.title("🚚 Simple Food Delivery Time Prediction")

    st.markdown("""
    This model predicts food delivery time based on:
    - **Distance (km)**
    - **Preparation Time (minutes)**
    - **Courier Experience (years)**
    """)

    # Input Panel
    st.header("Enter Parameters")

    distance = st.slider("Distance (km)", 0.5, 20.0, 5.0, 0.1)
    prep_time = st.slider("Preparation Time (minutes)", 5, 60, 15)
    courier_exp = st.slider("Courier Experience (years)", 0, 10, 2)

    predict_btn = st.button("Predict Delivery Time")

    if predict_btn:
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [courier_exp]
        })

        processed_input = preprocess_input(input_data, cols)
        processed_scaled = scaler.transform(processed_input)
        prediction = model.predict(processed_scaled)

        st.markdown("---")
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px">
            <h2 style="color:#2e86c1;text-align:center;">
            Predicted Delivery Time: {round(prediction[0], 1)} minutes
            </h2>
        </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        st.markdown("### 🔍 Top 5 Feature Importance")
        try:
            if hasattr(model, "coef_"):
                importance = model.coef_
            elif hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            else:
                importance = None

            if importance is not None:
                importance_df = pd.DataFrame({
                    "Feature": processed_input.columns,
                    "Importance": np.abs(importance)
                }).sort_values(by="Importance", ascending=False).head(5)

                # Show table
                st.dataframe(importance_df, use_container_width=True)

                # Show bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(
                    data=importance_df,
                    x="Importance",
                    y="Feature",
                    palette="Blues_d",
                    ax=ax
                )
                ax.set_title("Top 5 Important Features", fontsize=12)
                ax.set_xlabel("Coefficient Magnitude")
                ax.set_ylabel("")
                st.pyplot(fig)

            else:
                st.info("Model ini tidak mendukung feature importance.")
        except Exception as e:
            st.error(f"Gagal menampilkan feature importance: {e}")

if __name__ == "__main__":
    main()
