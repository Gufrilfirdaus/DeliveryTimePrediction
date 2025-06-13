import streamlit as st
st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="⏱️")  # Harus di paling atas

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
    for c in ['Weather', 'Traffic_Level', 'Vehicle_Type']:
        df[c].fillna(df[c].mode()[0], inplace=True)
    return df

model, scaler, cols = load_model()
df = load_data()

def preprocess_input(input_df):
    categorical_features = ['Weather', 'Traffic_Level', 'Vehicle_Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ],
        remainder='passthrough'
    )

    processed_data = preprocessor.fit_transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=feature_names)

    return processed_df

def main():
    st.title("Food Delivery Time Prediction")

    st.markdown("""
    Predict delivery time based on:
    - **Distance**
    - **Preparation Time**
    - **Courier Experience**
    - **Weather**, **Traffic**, and **Vehicle Type**
    """)

    st.header("Enter Delivery Parameters")
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1, format="%.1f")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=1, max_value=120, value=15, step=1)
        courier_exp = st.number_input("Courier Experience (years)", min_value=0, max_value=50, value=2, step=1)

    with col2:
        weather = st.selectbox("Weather Conditions", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        vehicle = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])

    predict_btn = st.button("Predict Delivery Time", type="primary")

    if predict_btn:
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [courier_exp],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Vehicle_Type': [vehicle]
        })

        processed_input = preprocess_input(input_data)
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
        st.markdown("### Top 5 Feature Importance")
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

                st.dataframe(importance_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_d", ax=ax)
                ax.set_title("Top 5 Important Features", fontsize=12)
                ax.set_xlabel("Coefficient Magnitude")
                ax.set_ylabel("")
                st.pyplot(fig)

            else:
                st.info("Model does not support feature importance.")
        except Exception as e:
            st.error(f"Failed to show feature importance: {e}")

if __name__ == "__main__":
    main()
