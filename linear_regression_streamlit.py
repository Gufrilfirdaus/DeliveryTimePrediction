import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("linear_regression_model.pkl")  # Pastikan file model.pkl tersedia di folder yang sama

# Judul Aplikasi
st.title("📦 Prediksi Waktu Pengiriman Makanan")

# Sidebar untuk Input User
st.sidebar.header("Masukkan Informasi Pengiriman")

# Input Kategorikal
weather = st.sidebar.selectbox("Kondisi Cuaca", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
traffic = st.sidebar.selectbox("Tingkat Kemacetan", ["Low", "Medium", "High"])
vehicle = st.sidebar.selectbox("Jenis Kendaraan", ["Bike", "Scooter", "Car"])
time_of_day = st.sidebar.selectbox("Waktu Pengiriman", ["Morning", "Afternoon", "Evening", "Night"])

# Input Numerikal
experience = st.sidebar.slider("Pengalaman Kurir (tahun)", 0, 20, 2)
distance = st.sidebar.slider("Jarak Tempuh (km)", 0.5, 50.0, 5.0, step=0.5)
prep_time = st.sidebar.slider("Waktu Persiapan Makanan (menit)", 0, 120, 15)

# Buat DataFrame Input
input_df = pd.DataFrame({
    'Courier_Experience_yrs': [experience],
    'Distance_km': [distance],
    'Preparation_Time_min': [prep_time],
    'Weather': [weather],
    'Traffic_Level': [traffic],
    'Vehicle_Type': [vehicle],
    'Time_of_Day': [time_of_day]
})

# One-hot Encoding
input_encoded = pd.get_dummies(input_df)

# Pastikan kolom sama seperti saat training
expected_cols = [
    'Courier_Experience_yrs', 'Distance_km', 'Preparation_Time_min',
    'Weather_Clear', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Foggy', 'Weather_Windy',
    'Traffic_Level_Low', 'Traffic_Level_Medium', 'Traffic_Level_High',
    'Vehicle_Type_Bike', 'Vehicle_Type_Scooter', 'Vehicle_Type_Car,
    'Time_of_Day_Morning', 'Time_of_Day_Afternoon', 'Time_of_Day_Evening', 'Time_of_Day_Night'
]
for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_cols]

# Prediksi
prediction = model.predict(input_encoded)[0]

# Output
st.subheader("🕒 Estimasi Waktu Pengiriman")
st.write(f"🚚 Perkiraan waktu pengiriman: **{prediction:.2f} menit**")

# Info Tambahan
st.markdown("---")
st.markdown("📊 Aplikasi ini mempertimbangkan variabel seperti cuaca, kemacetan, jenis kendaraan, dan pengalaman kurir untuk memperkirakan waktu pengiriman makanan.")
