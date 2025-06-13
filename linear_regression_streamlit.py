import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# — Page Config
st.set_page_config(
    page_title="Food Delivery Time Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .feature-importance {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .form-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# — Load Model, Scaler, Columns, Data
base = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base, "linear_reg_model.pkl"))
scaler = joblib.load(os.path.join(base, "scaler.pkl"))
cols = joblib.load(os.path.join(base, "linear_reg_model_columns.pkl"))
df = pd.read_csv(os.path.join(base, "Food_Delivery_Times.csv"))
df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)
for c in ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']:
    df[c].fillna(df[c].mode()[0], inplace=True)

# — Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This app predicts food delivery times based on various factors including:
    - Distance
    - Weather conditions
    - Traffic levels
    - Vehicle type
    - Courier experience
    - Preparation time
    """)

# — Tabs
tab_intro, tab_eda, tab_pred = st.tabs(["Introduction", "Data Analysis", "Prediction"])

# — Introduction
with tab_intro:
    st.header("Food Delivery Time Prediction")
    
    st.markdown("""
    ### Welcome to the Food Delivery Time Prediction App
    
    This application helps predict delivery times based on various factors that affect the delivery process.
    The model was trained on historical delivery data and can provide accurate estimates.
    """)

# — EDA
with tab_eda:
    st.header("Data Analysis")
    
    with st.expander("Dataset Overview", expanded=True):
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Descriptive Statistics")
        st.dataframe(df[['Delivery_Time_min','Distance_km','Preparation_Time_min',
                         'Courier_Experience_yrs','Traffic_Level']].describe().style.format("{:.2f}"))
    
    st.subheader("Feature Relationships with Delivery Time")
    
    tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
    
    with tab1:
        num_feat = st.selectbox("Select numerical feature:", 
                               ['Distance_km','Courier_Experience_yrs','Preparation_Time_min'])
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.scatterplot(df, x=num_feat, y='Delivery_Time_min', ax=ax)
        ax.set_title(f"Delivery Time vs {num_feat}")
        st.pyplot(fig)
        
    with tab2:
        cat_feat = st.selectbox("Select categorical feature:", 
                               ['Weather','Traffic_Level','Vehicle_Type'])
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(df, x=cat_feat, y='Delivery_Time_min', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"Delivery Time by {cat_feat}")
        st.pyplot(fig)

# — Prediction
with tab_pred:
    st.header("Delivery Time Prediction")
    
    # Input form
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Route Information")
            distance = st.selectbox("Distance (km)", 
                                  sorted(df['Distance_km'].unique()))
            traffic = st.selectbox("Traffic Level", 
                                 ['Low', 'Medium', 'High'])
            weather = st.selectbox("Weather Condition", 
                                 sorted(df['Weather'].unique()))
            
        with col2:
            st.subheader("Delivery Details")
            vehicle = st.selectbox("Vehicle Type", 
                                 sorted(df['Vehicle_Type'].unique()))
            experience = st.selectbox("Courier Experience (years)", 
                                    sorted(df['Courier_Experience_yrs'].unique()))
            prep_time = st.selectbox("Preparation Time (minutes)", 
                                   sorted(df['Preparation_Time_min'].unique()))
            time_of_day = st.selectbox("Time of Day", 
                                     sorted(df['Time_of_Day'].unique()))
        
        submitted = st.form_submit_button("Predict Delivery Time", 
                                        type="primary", 
                                        use_container_width=True)
    
    if submitted:
        # Prepare input data
        traffic_map = {'Low':0, 'Medium':1, 'High':2}
        data = {c:0 for c in cols}
        data.update({
            "Distance_km": distance,
            "Courier_Experience_yrs": experience,
            "Preparation_Time_min": prep_time,
            "Traffic_Level": traffic_map[traffic],
            f"Weather_{weather}": 1,
            f"Vehicle_Type_{vehicle}": 1,
            f"Time_of_Day_{time_of_day}": 1,
        })
        
        # Make prediction
        input_df = pd.DataFrame([data])[cols]
        x_scaled = scaler.transform(input_df)
        pred = model.predict(x_scaled)[0]
        avg = df["Delivery_Time_min"].mean()
        diff = pred - avg
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Estimated Delivery Time", 
                         f"{pred:.1f} minutes")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Delivery Time", 
                         f"{avg:.1f} minutes",
                         delta=f"{diff:+.1f} minutes")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("---")
        st.subheader("Feature Impact Analysis")
        
        coeffs = pd.DataFrame({"Feature": cols, "Coefficient": model.coef_})
        coeffs["Absolute_Impact"] = coeffs["Coefficient"].abs()
        top_features = coeffs.sort_values("Absolute_Impact", ascending=False).head(5)
        
        # Map feature names to more readable format
        feature_names = {
            "Distance_km": "Distance (km)",
            "Preparation_Time_min": "Prep Time (min)",
            "Traffic_Level": "Traffic Level",
            "Courier_Experience_yrs": "Courier Exp (yrs)",
            "Weather_Clear": "Weather: Clear",
            "Weather_Foggy": "Weather: Foggy",
            "Weather_Rainy": "Weather: Rainy",
            "Weather_Snowy": "Weather: Snowy",
            "Weather_Windy": "Weather: Windy",
            "Vehicle_Type_Bike": "Vehicle: Bike",
            "Vehicle_Type_Car": "Vehicle: Car",
            "Vehicle_Type_Scooter": "Vehicle: Scooter",
            "Time_of_Day_Morning": "Time: Morning",
            "Time_of_Day_Afternoon": "Time: Afternoon",
            "Time_of_Day_Evening": "Time: Evening",
            "Time_of_Day_Night": "Time: Night"
        }
        
        top_features["Feature"] = top_features["Feature"].map(feature_names).fillna(top_features["Feature"])
        
        st.dataframe(top_features[["Feature", "Coefficient"]].set_index("Feature")
                    .style.format("{:.2f}").background_gradient(cmap="RdBu", axis=0))
