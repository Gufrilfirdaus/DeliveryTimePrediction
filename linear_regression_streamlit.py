import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== LOAD DATA, MODEL, SCALER ====================
current_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(current_dir, "linear_reg_model.pkl"))
scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))  # ✅ Load scaler
model_columns = joblib.load(os.path.join(current_dir, "linear_reg_model_columns.pkl"))
df = pd.read_csv(os.path.join(current_dir, "Food_Delivery_Times.csv"))

# Preprocess for EDA
cat_cols = ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']
df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==================== STREAMLIT UI ===========================
st.title("🍔🚚 Food Delivery Time Analysis & Prediction")
st.markdown("Insights on delivery times + real-time predictions using your inputs.")

tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🔍 Feature Insights", "⏱️ Prediction"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.write(df.describe())
    fig, ax = plt.subplots()
    sns.histplot(df['Delivery_Time_min'], kde=True, bins=30, ax=ax)
    st.pyplot(fig)
    st.write(
        "**Insight:** Most deliveries fall between ~40–80 minutes. "
        "Long tails may be due to weather, traffic or distance."
    )

with tab2:
    st.header("Feature Impact Analysis")
    feature = st.selectbox(
        "Select feature:", 
        ['Distance_km','Preparation_Time_min','Courier_Experience_yrs',
         'Weather','Traffic_Level','Vehicle_Type','Time_of_Day']
    )
    fig, ax = plt.subplots()
    if feature in ['Distance_km','Preparation_Time_min','Courier_Experience_yrs']:
        sns.scatterplot(data=df, x=feature, y='Delivery_Time_min', ax=ax)
    else:
        sns.boxplot(data=df, x=feature, y='Delivery_Time_min', ax=ax)
        plt.xticks(rotation=45)
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write(
        "- ✅ `Distance` shows strongest correlation\n"
        "- 🚗 Longer prep, worse weather/traffic → slower deliveries"
    )

with tab3:
    st.header("Make a Prediction")
    st.markdown("Set the parameters below:")
    # Options
    weather_opts = sorted({c.split("_",1)[1] for c in model_columns if c.startswith("Weather_")})
    vehicle_opts = sorted({c.split("_",1)[1] for c in model_columns if c.startswith("Vehicle_Type_")})
    time_opts = sorted({c.split("_",1)[1] for c in model_columns if c.startswith("Time_of_Day_")})
    traffic_map = {'Low':0, 'Medium':1, 'High':2}
    traffic_opts = list(traffic_map.keys())

    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("Weather", weather_opts)
        traffic = st.selectbox("Traffic Level", traffic_opts)
        vehicle = st.selectbox("Vehicle Type", vehicle_opts)
    with col2:
        time_of_day = st.selectbox("Time of Day", time_opts)
        experience = st.number_input("Courier Experience (yrs)", 0, 30, 2)
        distance = st.number_input("Distance (km)", 0.0, 100.0, 5.0, 0.1)
        prep_time = st.number_input("Prep Time (mins)", 0, 60, 10)

    input_dict = {c:0 for c in model_columns}
    # Assign values
    input_dict["Distance_km"] = distance
    input_dict["Courier_Experience_yrs"] = experience
    input_dict["Preparation_Time_min"] = prep_time
    input_dict["Traffic_Level"] = traffic_map[traffic]
    input_dict[f"Weather_{weather}"] = 1
    input_dict[f"Vehicle_Type_{vehicle}"] = 1
    input_dict[f"Time_of_Day_{time_of_day}"] = 1

    input_df = pd.DataFrame([input_dict])[model_columns]

    if st.button("Predict Delivery Time"):
        st.write("🧾 Input Data:", input_df)
        scaled = scaler.transform(input_df)  # scale before predicting
        st.write("⚙️ Scaled Input:", scaled)
        pred = model.predict(scaled)[0]
        avg = df["Delivery_Time_min"].mean()
        diff = pred - avg
        st.success(f"⏱️ Estimated delivery time: {pred:.2f} mins")
        st.info(f"That's {abs(diff):.2f} minutes {'faster' if diff<0 else 'slower'} than average ({avg:.2f} mins)")

        imp = pd.DataFrame({
            "Feature": model_columns,
            "Coef": model.coef_
        }).sort_values("Coef", key=abs, ascending=False).head(5)
        st.subheader("Top Factors (coefficients):")
        st.dataframe(imp.set_index("Feature"))

        st.balloons()

# Footer
st.markdown("---")
st.write("Built with Streamlit • Model + Scaler loaded via joblib")
