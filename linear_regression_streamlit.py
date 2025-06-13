import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="⏱️")

# Load model and scaler
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(base, "linear_reg_model.pkl"))
        scaler = joblib.load(os.path.join(base, "scaler.pkl"))
        cols = joblib.load(os.path.join(base, "linear_reg_model_columns.pkl"))
        return model, scaler, cols
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

# Load dataset
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        df = pd.read_csv(os.path.join(base, "Food_Delivery_Times.csv"))
        df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)
        for c in ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']:
            df[c].fillna(df[c].mode()[0], inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

model, scaler, model_columns = load_model()
df = load_data()

# Preprocessing input
def preprocess_input(input_df, model_columns, scaler):
    input_df = input_df[['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
                         'Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']]

    categorical_features = ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']
    numerical_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories=[
                ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy'],
                ['Low', 'Medium', 'High'],
                ['Scooter', 'Bike', 'Car'],
                ['Morning', 'Afternoon', 'Evening', 'Night']
            ]), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    processed_data = preprocessor.fit_transform(input_df)

    if scaler is not None:
        processed_data = scaler.transform(processed_data)

    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(processed_data, columns=feature_names)

# Main App
def main():
    st.title("Food Delivery Time Prediction")

    st.markdown("""
    Predict delivery time based on:
    - Distance
    - Weather
    - Traffic
    - Vehicle
    - Courier Experience
    - Preparation Time
    - Time of Day
    """)

    st.header("Enter Delivery Parameters")
    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km)", min_value=0.1, value=5.0, step=0.1)
        prep_time = st.number_input("Preparation Time (minutes)", min_value=1, value=15, step=1)
        courier_exp = st.number_input("Courier Experience (years)", min_value=0.0, value=2.0, step=0.5)

    with col2:
        weather = st.selectbox("Weather Conditions", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        vehicle = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    if st.button("Predict Delivery Time", type="primary"):
        if model is None or scaler is None or model_columns is None:
            st.error("Model atau scaler tidak tersedia.")
            return

        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [courier_exp],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Vehicle_Type': [vehicle],
            'Time_of_Day': [time_of_day]
        })

        try:
            processed_input = preprocess_input(input_data, model_columns, scaler)
            prediction = model.predict(processed_input)

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

                    st.dataframe(importance_df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_d", ax=ax)
                    ax.set_title("Top 5 Important Features", fontsize=12)
                    ax.set_xlabel("Coefficient Magnitude")
                    ax.set_ylabel("")
                    st.pyplot(fig)
                else:
                    st.info("Model tidak mendukung perhitungan feature importance.")
            except Exception as e:
                st.error(f"Gagal menampilkan feature importance: {e}")

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

if __name__ == "__main__":
    main()
