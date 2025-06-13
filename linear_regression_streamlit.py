import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model (replace with your actual model file)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('linear_reg_model.pkl')  # Change to your model file
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'delivery_time_model.pkl' exists.")
        return None

# Preprocessing function (should match your training preprocessing)
def preprocess_input(input_df):
    # Define categorical columns (should match your training setup)
    categorical_features = ['Weather', 'Traffic_Level', 'Vehicle_Type']
    
    # Create transformer (should match your training setup)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform (in a real app, you should pre-fit the encoder during training)
    processed_data = preprocessor.fit_transform(input_df)
    return processed_data

# Main app function
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
    
    # Sidebar with input controls
    with st.sidebar:
        st.header("Input Parameters")
        
        distance = st.slider("Distance (km)", 0.5, 20.0, 5.0, 0.1)
        prep_time = st.slider("Preparation Time (minutes)", 5, 30, 15)
        courier_exp = st.slider("Courier Experience (years)", 0, 10, 2)
        
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
        
        predict_btn = st.button("Predict Delivery Time")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("How It Works")
        st.markdown("""
        This predictive model uses machine learning to estimate food delivery times 
        based on historical data and key factors that affect delivery duration.
        
        The model considers:
        - Route characteristics (distance, traffic)
        - Environmental conditions (weather)
        - Delivery resources (vehicle type, courier experience)
        - Restaurant preparation time
        """)
        
        st.image("https://images.unsplash.com/photo-1585032226651-759b368d7246?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
                caption="Food Delivery Analysis")
    
    # Prediction logic
    if predict_btn:
        model = load_model()
        if model is not None:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'Distance_km': [distance],
                'Preparation_Time_min': [prep_time],
                'Courier_Experience_yrs': [courier_exp],
                'Weather': [weather],
                'Traffic_Level': [traffic],
                'Vehicle_Type': [vehicle]
            })
            
            # Preprocess the input
            processed_input = preprocess_input(input_data)
            
            # Make prediction
            prediction = model.predict(processed_input)
            
            # Display result
            st.success(f"### Predicted Delivery Time: {round(prediction[0], 1)} minutes")
            
            # Show interpretation
            st.info("""
            **Interpretation Tips:**
            - Times under 30 minutes: Excellent service
            - 30-45 minutes: Normal delivery range  
            - 45-60 minutes: Slightly delayed
            - Over 60 minutes: Consider optimizing operations
            """)
    
    # Footer
    st.markdown("---")
    st.caption("""
    *Note: Predictions are estimates based on historical data. Actual delivery times may vary.*
    """)

if __name__ == "__main__":
    main()
