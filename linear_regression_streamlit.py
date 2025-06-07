import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== LOAD DATA AND MODEL ====================
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to model and data files
model_path = os.path.join(current_dir, "linear_reg_model.pkl")
model_columns_path = os.path.join(current_dir, "linear_reg_model_columns.pkl")
data_path = os.path.join(current_dir, "Food_Delivery_Times.csv")

# Load model, columns, and original data
try:
    model = joblib.load(model_path)
    model_columns = joblib.load(model_columns_path)
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.stop()

# Preprocess data for EDA
df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())
cat_cols = ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# =============================================================

# ==================== STREAMLIT UI ===========================
st.title("🍔🚚 Food Delivery Time Analysis & Prediction")
st.markdown("""
This app provides insights into food delivery times and predicts delivery duration based on various factors.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🔍 Feature Insights", "⏱️ Delivery Time Prediction"])

with tab1:
    st.header("Exploratory Data Analysis")
    
    # Basic statistics
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(df)}")
    st.write(df.describe())
    
    # Delivery Time Distribution
    st.subheader("Delivery Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Delivery_Time_min'], kde=True, bins=30, ax=ax)
    ax.set_xlabel('Delivery Time (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Delivery Times')
    st.pyplot(fig)
    
    st.write("""
    **Insight:** The delivery time distribution shows most deliveries take between 40-80 minutes, 
    with some outliers taking significantly longer. This could be due to extreme weather conditions, 
    heavy traffic, or longer distances.
    """)

with tab2:
    st.header("Feature Impact Analysis")
    
    # Feature vs Delivery Time
    st.subheader("How Features Affect Delivery Time")
    feature = st.selectbox("Select feature to analyze:", 
                         ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 
                          'Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day'])
    
    fig, ax = plt.subplots()
    
    if feature in ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']:
        sns.scatterplot(data=df, x=feature, y='Delivery_Time_min', ax=ax)
        ax.set_title(f'Delivery Time vs {feature}')
    else:
        sns.boxplot(data=df, x=feature, y='Delivery_Time_min', ax=ax)
        ax.set_title(f'Delivery Time by {feature}')
        plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("""
    **Key Insights:**
    - Distance has the strongest correlation with delivery time (as expected)
    - Courier experience shows a slight negative correlation with delivery time
    - Preparation time has minimal direct impact on delivery time
    - Weather and traffic conditions significantly affect delivery times
    """)

with tab3:
    st.header("Delivery Time Prediction")
    st.markdown("Enter delivery details to estimate the delivery time:")
    
    # Get categories from model_columns
    weather_options = sorted({col.replace("Weather_", "") for col in model_columns if col.startswith("Weather_")})
    traffic_options = sorted({col.replace("Traffic_Level_", "") for col in model_columns if col.startswith("Traffic_Level_")})
    vehicle_options = sorted({col.replace("Vehicle_Type_", "") for col in model_columns if col.startswith("Vehicle_Type_")})
    time_options = sorted({col.replace("Time_of_Day_", "") for col in model_columns if col.startswith("Time_of_Day_")})
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        weather = st.selectbox("Weather", weather_options)
        traffic = st.selectbox("Traffic Level", traffic_options)
        vehicle = st.selectbox("Vehicle Type", vehicle_options)
        
    with col2:
        time_of_day = st.selectbox("Time of Day", time_options)
        experience = st.number_input("Courier Experience (years)", min_value=0, max_value=30, value=2)
        distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=5.0, step=0.1)
        prep_time = st.number_input("Food Preparation Time (minutes)", min_value=0, value=10)
    
    # Prepare input for model
    input_dict = {col: 0 for col in model_columns}
    
    # Fill numerical features
    input_dict["Courier_Experience_yrs"] = experience
    input_dict["Distance_km"] = distance
    input_dict["Preparation_Time_min"] = prep_time
    
    # Fill one-hot encoded categorical features
    input_dict[f"Weather_{weather}"] = 1
    input_dict[f"Traffic_Level_{traffic}"] = 1
    input_dict[f"Vehicle_Type_{vehicle}"] = 1
    input_dict[f"Time_of_Day_{time_of_day}"] = 1
    
    # Convert to DataFrame and arrange columns
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]
    
    if st.button("Predict Delivery Time"):
        try:
            prediction = model.predict(input_df)[0]
            
            # Show prediction with context
            st.success(f"⏱️ Estimated delivery time: **{prediction:.2f} minutes**")
            
            # Add context about the prediction
            avg_time = df['Delivery_Time_min'].mean()
            diff = prediction - avg_time
            
            if prediction < avg_time:
                st.info(f"📊 This is {abs(diff):.2f} minutes faster than average ({avg_time:.2f} minutes)")
            else:
                st.warning(f"📊 This is {diff:.2f} minutes slower than average ({avg_time:.2f} minutes)")
            
            # Show feature importance explanation
            st.subheader("Key Factors Affecting This Prediction:")
            
            # Get model coefficients (assuming linear regression)
            try:
                coefficients = model.coef_
                feature_names = model_columns
                
                # Create a DataFrame of feature importances
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Impact': coefficients
                }).sort_values('Impact', ascending=False)
                
                # Filter only the features that were set to 1 in our input
                active_features = [k for k, v in input_dict.items() if v == 1 and k != 'Preparation_Time_min']
                active_features.extend(['Distance_km', 'Courier_Experience_yrs', 'Preparation_Time_min'])
                
                # Show top impacting features for this prediction
                st.write("Most significant factors for this prediction:")
                st.dataframe(
                    importance_df[importance_df['Feature'].isin(active_features)]
                    .sort_values('Impact', key=abs, ascending=False)
                    .head(5)
                )
                
            except AttributeError:
                st.write("(Feature importance not available for this model type)")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Add footer
st.markdown("---")
st.markdown("""
**About this app:**
- Uses a machine learning model trained on historical delivery data
- Provides insights into factors affecting delivery times
- Helps optimize delivery operations by understanding key variables
""")
