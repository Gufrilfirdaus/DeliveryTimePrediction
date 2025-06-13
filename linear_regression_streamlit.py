import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======= Page Config & Theme =======
st.set_page_config(
    page_title="Food Delivery Time Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======= Load Model + Scaler + Data =======
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, "linear_reg_model.pkl"))
scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))
model_columns = joblib.load(os.path.join(current_dir, "linear_reg_model_columns.pkl"))
df = pd.read_csv(os.path.join(current_dir, "Food_Delivery_Times.csv"))

df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())
for c in ['Weather','Traffic_Level','Vehicle_Type','Time_of_Day']:
    df[c] = df[c].fillna(df[c].mode()[0])

# ======= Sidebar Inputs =======
with st.sidebar:
    st.header("Prediction Inputs")
    weather_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Weather_")])
    vehicle_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Vehicle_Type_")])
    time_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Time_of_Day_")])
    traffic_map = {'Low':0, 'Medium':1, 'High':2}

    weather = st.selectbox("Weather", weather_opts)
    traffic = st.selectbox("Traffic Level", list(traffic_map.keys()))
    vehicle = st.selectbox("Vehicle Type", vehicle_opts)
    time_of_day = st.selectbox("Time of Day", time_opts)
    experience = st.number_input("Courier Experience (yrs)", 0, 30, 2)
    distance = st.number_input("Distance (km)", 0.0, 100.0, 5.0, 0.1)
    prep_time = st.number_input("Preparation Time (mins)", 0, 60, 10)
    if st.button("Predict Delivery Time"):
        input_dict = {c: 0 for c in model_columns}
        input_dict.update({
            "Distance_km": distance,
            "Courier_Experience_yrs": experience,
            "Preparation_Time_min": prep_time,
            "Traffic_Level": traffic_map[traffic],
            f"Weather_{weather}": 1,
            f"Vehicle_Type_{vehicle}": 1,
            f"Time_of_Day_{time_of_day}": 1,
        })
        input_df = pd.DataFrame([input_dict])[model_columns]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        avg = df["Delivery_Time_min"].mean()
        diff = pred - avg

        st.subheader("🚚 Estimated Delivery Time")
        st.metric(label="Duration (mins)", value=f"{pred:.0f}", delta=f"{diff:+.0f} vs avg")
        coeffs = pd.DataFrame({
            "Feature": model_columns,
            "Coef": model.coef_
        }).set_index("Feature").nlargest(5, columns="Coef", key=abs)
        st.subheader("Top Influencing Features (coef magnitude)")
        st.table(coeffs)

# ======= Main Layout =======

st.title("Food Delivery Time Dashboard")
st.markdown("Explore delivery time patterns and workflow insights backed by a trained ML model.")

tab1, tab2 = st.tabs(["📊 Data Analysis", "🔍 Feature Insights"])

with tab1:
    st.header("Delivery Time Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Delivery_Time_min'], kde=True, bins=30, ax=ax)
        ax.set_xlabel("Delivery Time (min)")
        st.pyplot(fig)
    with col2:
        st.write("**Dataset Overview**")
        st.write(df[['Delivery_Time_min','Distance_km','Preparation_Time_min','Courier_Experience_yrs','Traffic_Level']].describe())

with tab2:
    st.header("Correlation & Feature Effects")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("### 🚀 Impact of Key Features")
    for feat in ["Distance_km","Courier_Experience_yrs","Preparation_Time_min"]:
        fig, ax = plt.subplots()
        sns.scatterplot(df, x=feat, y="Delivery_Time_min", ax=ax)
        ax.set_title(f"Delivery Time vs {feat}")
        st.pyplot(fig)

    st.markdown("### 📦 Categorical Effects")
    for cat in ["Weather","Traffic_Level","Vehicle_Type","Time_of_Day"]:
        fig, ax = plt.subplots()
        sns.boxplot(df, x=cat, y="Delivery_Time_min", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with Python, Streamlit, and scikit‑learn. Trained model + scaler via joblib.")
