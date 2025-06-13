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

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .form-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# — Load Model and Data
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

# — Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    Predict food delivery times based on:
    - Distance
    - Weather conditions
    - Traffic levels
    - Vehicle type
    - Courier experience
    - Preparation time
    """)

# — Main Content
st.header("Delivery Time Prediction")

# Input form
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Route Information")
        distance = st.number_input("Distance (km)", 
                                 min_value=0.5, 
                                 max_value=50.0, 
                                 value=5.0, 
                                 step=0.5)
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
        prep_time = st.number_input("Preparation Time (minutes)", 
                          min_value=0,
                          max_value=120,
                          value=int(df['Preparation_Time_min'].median()),
                          step=1)       
        time_of_day = st.selectbox("Time of Day", 
                                 sorted(df['Time_of_Day'].unique()))
    
    submitted = st.form_submit_button("Predict Delivery Time", 
                                    type="primary")

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
    
    # Display results
    st.markdown("---")
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Estimated Delivery Time", 
                 f"{pred:.1f} minutes")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Delivery Time", 
                 f"{avg:.1f} minutes",
                 delta=f"{pred-avg:+.1f} minutes")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("---")
    st.subheader("Top Influencing Factors")
    
    coeffs = pd.DataFrame({"Feature": cols, "Impact": model.coef_})
    coeffs["Absolute_Impact"] = coeffs["Impact"].abs()
    top_features = coeffs.sort_values("Absolute_Impact", ascending=False).head(5)
    
    # Format feature names
    feature_names = {
        "Distance_km": "Distance (km)",
        "Preparation_Time_min": "Prep Time (min)",
        "Traffic_Level": "Traffic Level",
        "Courier_Experience_yrs": "Courier Exp (yrs)",
        **{f"Weather_{w}": f"Weather: {w}" for w in df['Weather'].unique()},
        **{f"Vehicle_Type_{v}": f"Vehicle: {v}" for v in df['Vehicle_Type'].unique()},
        **{f"Time_of_Day_{t}": f"Time: {t}" for t in df['Time_of_Day'].unique()}
    }
    
    top_features["Feature"] = top_features["Feature"].map(feature_names)
    st.dataframe(top_features[["Feature", "Impact"]].set_index("Feature")
                .style.format("{:.2f}").background_gradient(cmap="RdBu", axis=0))
