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
weather = st.sidebar.selectbox("Kondisi Cuaca", ["Clear", "Rainy", "Fog", "Snow"])
traffic = st.sidebar.selectbox("Tingkat Kemacetan", ["Low", "Medium", "High", "Jam"])
vehicle = st.sidebar.selectbox("Jenis Kendaraan", ["Motorcycle", "Car", "Bicycle"])
time_of_day = st.sidebar.selectbox("Waktu Pengiriman", ["Pagi", "Siang", "Sore", "Malam"])

# Input Numerikal
experience = st.sidebar.slider("Pengalaman Kurir (tahun)", 0, 20, 2)
distance = st.sidebar.slider("Jarak Tempuh (km)", 0.5, 50.0, 5.0, step=0.5)
prep_time = st.sidebar.slider("Waktu Persiapan Makanan (menit)", 0, 120, 15)

# Buat DataFrame Input
input_df = pd.DataFrame({
    'courier_experience_yrs': [experience],
    'Distance_km': [distance],
    'Preparation_Time_min': [prep_time],
    'weather': [weather],
    'traffic_level': [traffic],
    'vehicle_type': [vehicle],
    'time_of_day': [time_of_day]
})

# One-hot Encoding
input_encoded = pd.get_dummies(input_df)

# Pastikan kolom sama seperti saat training
expected_cols = [
    'courier_experience_yrs', 'Distance_km', 'Preparation_Time_min',
    'weather_Clear', 'weather_Fog', 'weather_Rainy', 'weather_Snow',
    'traffic_level_High', 'traffic_level_Jam', 'traffic_level_Low', 'traffic_level_Medium',
    'vehicle_type_Bicycle', 'vehicle_type_Car', 'vehicle_type_Motorcycle',
    'time_of_day_Malam', 'time_of_day_Pagi', 'time_of_day_Siang', 'time_of_day_Sore'
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
