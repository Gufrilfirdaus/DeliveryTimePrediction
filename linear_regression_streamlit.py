import streamlit as st
st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="⏱️")

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load model, scaler, and columns
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
    for c in ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']:
        df[c].fillna(df[c].mode()[0], inplace=True)
    return df

model, scaler, cols = load_model()
df = load_data()

# Preprocessing function
def preprocess_input(input_df, reference_columns):
    categorical_features = ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    dummy_df = df[categorical_features + ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']].dropna()
    preprocessor.fit(dummy_df)
    
    processed_data = preprocessor.transform(input_df)
    encoded_feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data,
        columns=encoded_feature_names
    )

    for col in reference_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[reference_columns]
    
    return processed_df

def main():
    st.title("Food Delivery Time Prediction")
    st.markdown("""
    Predict delivery time based on:
    - **Distance**
    - **Weather conditions**
    - **Traffic levels**
    - **Vehicle type**
    - **Courier experience**
    - **Preparation time**
    - **Time of Day**
    """)

    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    This predictive model uses machine learning to estimate food delivery times 
    based on historical data and key factors that affect delivery duration.
    """)

    st.markdown("---")
    st.header("Enter Delivery Parameters")
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, format="%.1f")
        prep_time = st.number_input("Preparation Time (minutes)", min_value=1, max_value=120, value=15, step=1)
        courier_exp = st.number_input("Courier Experience (years)", min_value=0, max_value=50, value=2, step=1)

    with col2:
        weather = st.selectbox("Weather Conditions", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        vehicle = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    predict_btn = st.button("Predict Delivery Time", type="primary")

    if predict_btn:
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [courier_exp],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Vehicle_Type': [vehicle],
            'Time_of_Day': [time_of_day]
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

        # Feature Importance Section
        st.header("Feature Importance")
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
                }).sort_values(by="Importance", ascending=False)

                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("Model ini tidak mendukung perhitungan feature importance.")
        except Exception as e:
            st.error(f"Gagal menampilkan feature importance: {e}")

if __name__ == "__main__":
    main()
