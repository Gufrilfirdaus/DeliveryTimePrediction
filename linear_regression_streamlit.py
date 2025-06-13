import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# — Page Configuration
st.set_page_config(page_title="🚚 Delivery Time App", page_icon="🚚", layout="wide")

# — Load Model, Scaler, Columns, Data
base = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base, "linear_reg_model.pkl"))
scaler = joblib.load(os.path.join(base, "scaler.pkl"))
cols = joblib.load(os.path.join(base, "linear_reg_model_columns.pkl"))
df = pd.read_csv(os.path.join(base, "Food_Delivery_Times.csv"))
for c in ['Courier_Experience_yrs']:
    df[c].fillna(df[c].median(), inplace=True)
for c in ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']:
    df[c].fillna(df[c].mode()[0], inplace=True)

# — Tabs: Introduction, EDA, Prediction
tab_intro, tab_eda, tab_pred = st.tabs(["Introduction", "EDA", "Prediction"])

with tab_intro:
    st.header("🚀 Introduction")
    st.markdown("""
    Explore the food delivery dataset, check insights via EDA, and predict delivery time with your custom inputs.
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

    st.subheader("Scatter Plots")
    for feat in ['Distance_km','Courier_Experience_yrs','Preparation_Time_min']:
        fig, ax = plt.subplots()
        sns.scatterplot(df, x=feat, y='Delivery_Time_min', ax=ax)
        ax.set_title(f"Delivery Time vs {feat}")
        st.pyplot(fig)

    st.subheader("Boxplots for Categorical Features")
    for cat in ['Weather','Traffic_Level','Vehicle_Type','Time_of_Day']:
        fig, ax = plt.subplots()
        sns.boxplot(df, x=cat, y='Delivery_Time_min', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df['Delivery_Time_min'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

with tab_pred:
    st.header("⏱️ Predict Delivery Time")
    st.markdown("Select parameters below and click **Predict**.")
    
    # Input form
    with st.form("prediction_form"):
        cols = st.columns(3)
        weather_opts = sorted({c.split("_",1)[1] for c in cols if c.startswith("Weather_") for cols in [cols]})
        vehicle_opts = sorted({c.split("_",1)[1] for c in cols if c.startswith("Vehicle_Type_") for cols in [cols]})
        time_opts = sorted({c.split("_",1)[1] for c in cols if c.startswith("Time_of_Day_") for cols in [cols]})
        traffic_map = {'Low':0, 'Medium':1, 'High':2}

        col1, col2, col3 = st.columns(3)
        with col1:
            weather = st.selectbox("Weather", weather_opts)
            traffic = st.selectbox("Traffic Level", list(traffic_map.keys()))
        with col2:
            vehicle = st.selectbox("Vehicle Type", vehicle_opts)
            time_of_day = st.selectbox("Time of Day", time_opts)
        with col3:
            experience = st.number_input("Courier Experience (yrs)", 0, 30, 2)
            distance = st.number_input("Distance (km)", 0.0, 100.0, 5.0, 0.1)
            prep_time = st.number_input("Prep Time (min)", 0, 60, 10)
        
        submit = st.form_submit_button("Predict")

    if submit:
        data = {c: 0 for c in cols}
        data.update({
            "Distance_km": distance,
            "Courier_Experience_yrs": experience,
            "Preparation_Time_min": prep_time,
            "Traffic_Level": traffic_map[traffic],
            f"Weather_{weather}": 1,
            f"Vehicle_Type_{vehicle}": 1,
            f"Time_of_Day_{time_of_day}": 1,
        })
        input_df = pd.DataFrame([data])[cols]
        x_scaled = scaler.transform(input_df)
        pred = model.predict(x_scaled)[0]
        avg = df["Delivery_Time_min"].mean()
        diff = pred - avg

        st.metric("Estimated Delivery Time (min)", f"{pred:.1f}", delta=f"{diff:+.1f}")

        coeffs = pd.DataFrame({"Feature": cols, "Coef": model.coef_}).set_index("Feature")
        top = coeffs["Coef"].abs().nlargest(5).to_frame()
        st.subheader("Top Influencing Features")
        st.table(top)

st.markdown("---")
st.caption("Built with Streamlit — integrated model + scaler for accurate, user-friendly predictions")
