import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Preprocessing function
def preprocess_input(input_df):
    categorical_features = ['Weather', 'Traffic_Level', 'Vehicle_Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    processed_data = preprocessor.fit_transform(input_df)
    return processed_data

def main():
    st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="⏱️")
    
    # Header
    st.title("🍔 Food Delivery Time Prediction")
    st.markdown("""
    Predict delivery time based on:
    - **Distance** between restaurant and delivery location
    - Current **weather conditions**
    - Road **traffic levels**
    - Delivery **vehicle type**
    - **Courier experience**
    - **Preparation time**
    """)
    
    # Input parameters in main content
    st.header("Enter Delivery Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        distance = st.slider("Distance (km)", 0.5, 20.0, 5.0, 0.1)
        prep_time = st.slider("Preparation Time (minutes)", 5, 30, 15)
        courier_exp = st.slider("Courier Experience (years)", 0, 10, 2)
    
    with col2:
        weather = st.selectbox(
            "Weather Conditions",
            ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
        )
        
        traffic = st.selectbox(
            "Traffic Level",
            ["Low", "Medium", "High"]
        )
        
        vehicle = st.selectbox(
            "Vehicle Type",
            ["Scooter", "Bike", "Car"]
        )
    
    predict_btn = st.button("Predict Delivery Time", type="primary")
    
    # How It Works section
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    This predictive model uses machine learning to estimate food delivery times 
    based on historical data and key factors that affect delivery duration.
    """)
    
    # Prediction logic
    if predict_btn:
        model = load_model()
        if model is not None:
            input_data = pd.DataFrame({
                'Distance_km': [distance],
                'Preparation_Time_min': [prep_time],
                'Courier_Experience_yrs': [courier_exp],
                'Weather': [weather],
                'Traffic_Level': [traffic],
                'Vehicle_Type': [vehicle]
            })
            
            processed_input = preprocess_input(input_data)
            prediction = model.predict(processed_input)
            
            # Display prediction with visual emphasis
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px">
                <h2 style="color:#2e86c1;text-align:center;">
                Predicted Delivery Time: {round(prediction[0], 1)} minutes
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation guide
            st.markdown("### 📊 Interpretation Guide")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Fast Delivery", "< 30 min", 
                         help="Excellent service timeframe")
                
            with col_b:
                st.metric("Normal Delivery", "30-45 min",
                         help="Standard delivery range")
                
            with col_c:
                st.metric("Delayed Delivery", "> 45 min",
                         help="Consider optimizing operations")

if __name__ == "__main__":
    main()
