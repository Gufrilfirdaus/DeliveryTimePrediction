import os
import streamlit as st
import pandas as pd
import joblib

# ================= PERBAIKAN PATH =================
# Dapatkan path absolut ke direktori skrip
current_dir = os.path.dirname(os.path.abspath(__file__))

# Bangun path lengkap untuk file model
model_path = os.path.join(current_dir, "linear_reg_model.pkl")
model_columns_path = os.path.join(current_dir, "linear_reg_model_columns.pkl")

# Load model dengan error handling
try:
    model = joblib.load(model_path)
    model_columns = joblib.load(model_columns_path)
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memuat model: {str(e)}")
    st.error("Pastikan file model (.pkl) ada di direktori yang sama dengan skrip ini")
    st.stop()
# ==================================================

st.title("Prediksi Waktu Pengiriman Makanan (Linear Regression) 🚚")
st.markdown("Masukkan informasi berikut untuk memprediksi estimasi waktu pengiriman makanan:")

# Input dari user
weather = st.selectbox("Cuaca", ["Sunny", "Rainy", "Cloudy", "Snowy"])
traffic = st.selectbox("Tingkat Lalu Lintas", ["Low", "Medium", "High"])
vehicle = st.selectbox("Jenis Kendaraan", ["Bike", "Motorbike", "Car"])
time_of_day = st.selectbox("Waktu Pengiriman", ["Morning", "Afternoon", "Evening", "Night"])
experience = st.number_input("Pengalaman Kurir (tahun)", min_value=0, max_value=30, value=2)

# Inisialisasi dictionary input dengan semua kolom = 0
input_dict = {col: 0 for col in model_columns}

# Isi berdasarkan input user
input_dict["Courier_Experience_yrs"] = experience  # Perhatikan penulisan kolom!
input_dict[f"Weather_{weather}"] = 1
input_dict[f"Traffic_Level_{traffic}"] = 1
input_dict[f"Vehicle_Type_{vehicle}"] = 1
input_dict[f"Time_of_Day_{time_of_day}"] = 1

# Buat DataFrame input
input_df = pd.DataFrame([input_dict])

# Pastikan urutan kolom sama dengan saat training
input_df = input_df[model_columns]

# Prediksi
if st.button("Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimasi waktu pengiriman: **{prediction:.2f} menit**")
        st.balloons()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
