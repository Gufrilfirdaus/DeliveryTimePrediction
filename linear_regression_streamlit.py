import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ——————————— Page Config ———————————
st.set_page_config(
    page_title="🚚 Food Delivery Time Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ————————— Load Model & Data —————————
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, "linear_reg_model.pkl"))
scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))
model_columns = joblib.load(os.path.join(current_dir, "linear_reg_model_columns.pkl"))
df = pd.read_csv(os.path.join(current_dir, "Food_Delivery_Times.csv"))

for col in ['Courier_Experience_yrs']:
    df[col].fillna(df[col].median(), inplace=True)
for col in ['Weather','Traffic_Level','Vehicle_Type','Time_of_Day']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ————————— Sidebar (Prediction Input) —————————
st.sidebar.header("🧮 Prediction Input")
weather_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Weather_")])
vehicle_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Vehicle_Type_")])
time_opts = sorted([c.split("_",1)[1] for c in model_columns if c.startswith("Time_of_Day_")])
traffic_map = {'Low':0, 'Medium':1, 'High':2}

weather = st.sidebar.selectbox("Weather", weather_opts)
traffic = st.sidebar.selectbox("Traffic Level", list(traffic_map.keys()))
vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_opts)
time_of_day = st.sidebar.selectbox("Time of Day", time_opts)
experience = st.sidebar.number_input("Courier Experience (yrs)", 0, 30, 2)
distance = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 5.0, 0.1)
prep_time = st.sidebar.number_input("Preparation Time (min)", 0, 60, 10)
predict_btn = st.sidebar.button("Predict Delivery Time")

# ————————— Main Tabs —————————
st.title("🚚 Food Delivery Time Dashboard")
tab_intro, tab_eda, tab_pred = st.tabs(["Introduction", "EDA", "Prediction"])

with tab_intro:
    st.header("🔎 Introduction")
    st.markdown("""
    Welcome! This dashboard allows you to:
    1. **Explore** historical delivery data with descriptive stats & visualizations.  
    2. **Make predictions** of delivery time using your own inputs.  
    The model is trained with Linear Regression on scaled features.
    """)

with tab_eda:
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Descriptive Statistics")
    st.write(df[['Delivery_Time_min','Distance_km','Preparation_Time_min',
                 'Courier_Experience_yrs','Traffic_Level']].describe())

    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Scatter Plot")
    for feat in ['Distance_km','Courier_Experience_yrs','Preparation_Time_min']:
        fig, ax = plt.subplots()
        sns.scatterplot(df, x=feat, y='Delivery_Time_min', ax=ax)
        ax.set_title(f"Delivery Time vs {feat}")
        st.pyplot(fig)

    st.subheader("Boxplot")
    for cat in ['Weather','Traffic_Level','Vehicle_Type','Time_of_Day']:
        fig, ax = plt.subplots()
        sns.boxplot(df, x=cat, y='Delivery_Time_min', ax=ax)
        ax.set_title(f"Delivery Time by {cat}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df['Delivery_Time_min'], bins=30, kde=True, ax=ax)
    ax.set_title("Delivery Time Distribution")
    st.pyplot(fig)

with tab_pred:
    st.header("⏱️ Prediction")
    if predict_btn:
        input_dict = {c:0 for c in model_columns}
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
        delta = pred - avg

        st.metric(label="Estimated Delivery Time (min)",
                  value=f"{pred:.1f}",
                  delta=f"{delta:+.1f} vs avg")

        coeffs = pd.DataFrame({
            "Feature": model_columns,
            "Coef": model.coef_
        }).set_index("Feature")
        st.subheader("Top Influencing Features")
        st.table(coeffs['Coef'].abs().nlargest(5).rename_axis("Feature").to_frame())

st.markdown("---")
st.caption("Built with Streamlit — layout using sidebar & tabs :contentReference[oaicite:1]{index=1}")
