import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# — Page Config
st.set_page_config(
    page_title="🚚 Food Delivery Time Prediction", 
    page_icon="🚚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        padding: 1rem 0;
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
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
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
    This app predicts food delivery times based on:
    - Distance
    - Weather conditions
    - Traffic levels
    - Vehicle type
    - Courier experience
    - Preparation time
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Model Information
    - **Algorithm**: Linear Regression
    - **R² Score**: 0.92 (on test set)
    - **MAE**: 6.3 minutes
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# — Tabs
tab_intro, tab_eda, tab_pred = st.tabs(["📌 Introduction", "📊 Data Analysis", "⏱️ Predict Delivery Time"])

# — Introduction
with tab_intro:
    st.header("Food Delivery Time Prediction", divider="blue")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        ### Welcome to the Food Delivery Time Prediction App
        
        This application helps food delivery businesses and customers understand and predict delivery times 
        based on various factors that affect the delivery process. The model was trained on historical 
        delivery data and can provide accurate estimates to improve operational efficiency.
        
        #### Key Features:
        - **Exploratory Data Analysis**: Visualize relationships between delivery time and various factors
        - **Delivery Time Prediction**: Get instant estimates based on current conditions
        - **Feature Importance**: Understand which factors most impact delivery times
        """)
    
    with col2:
        # You can replace this with your own image if available
        st.image("https://cdn-icons-png.flaticon.com/512/3712/3712476.png", 
                 width=200, caption="Delivery Time Prediction")
    
    st.markdown("---")
    
    st.subheader("Problem Background")
    st.markdown("""
    In the food delivery service industry, speed and delivery time accuracy are key factors in maintaining 
    customer satisfaction and business competitiveness. Many factors can affect delivery duration, such as:
    - Weather conditions
    - Traffic congestion levels
    - Time of day
    - Vehicle type used
    - Courier experience
    
    Therefore, leveraging historical data to predict delivery time more accurately becomes very important, 
    especially in efforts to improve operational efficiency and strategic decision making.
    """)

# — EDA
with tab_eda:
    st.header("Exploratory Data Analysis", divider="blue")
    
    with st.expander("🔍 Dataset Overview", expanded=True):
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), height=300)
        
        st.subheader("Descriptive Statistics")
        st.dataframe(df[['Delivery_Time_min','Distance_km','Preparation_Time_min',
                         'Courier_Experience_yrs','Traffic_Level']].describe().style.format("{:.2f}"))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📈 Correlation Analysis"):
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df.select_dtypes(include=np.number).corr(), 
                        annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Between Numerical Features")
            st.pyplot(fig)
            
    with col2:
        with st.expander("📊 Delivery Time Distribution"):
            fig, ax = plt.subplots()
            sns.histplot(df['Delivery_Time_min'], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Delivery Times")
            st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Feature Relationships with Delivery Time")
    
    tab1, tab2, tab3 = st.tabs(["Numerical Features", "Categorical Features", "Time of Day Analysis"])
    
    with tab1:
        st.markdown("### Numerical Features vs Delivery Time")
        num_feat = st.selectbox("Select numerical feature:", 
                               ['Distance_km','Courier_Experience_yrs','Preparation_Time_min'])
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.scatterplot(df, x=num_feat, y='Delivery_Time_min', ax=ax)
        ax.set_title(f"Delivery Time vs {num_feat}", fontsize=14)
        st.pyplot(fig)
        
    with tab2:
        st.markdown("### Categorical Features vs Delivery Time")
        cat_feat = st.selectbox("Select categorical feature:", 
                               ['Weather','Traffic_Level','Vehicle_Type'])
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(df, x=cat_feat, y='Delivery_Time_min', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"Delivery Time by {cat_feat}", fontsize=14)
        st.pyplot(fig)
        
    with tab3:
        st.markdown("### Delivery Time Patterns by Time of Day")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        
        # Boxplot
        sns.boxplot(df, x='Time_of_Day', y='Delivery_Time_min', ax=ax1)
        ax1.set_title("Delivery Time Distribution by Time of Day")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Countplot
        sns.countplot(df, x='Time_of_Day', ax=ax2, order=df['Time_of_Day'].value_counts().index)
        ax2.set_title("Number of Deliveries by Time of Day")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        st.pyplot(fig)

# — Prediction
with tab_pred:
    st.header("Delivery Time Prediction", divider="blue")
    
    st.markdown("""
    Fill in the form below to get a delivery time estimate. The model considers:
    - Distance to destination
    - Current weather conditions
    - Traffic levels
    - Vehicle type being used
    - Courier experience
    - Food preparation time
    """)
    
    # Input form
    with st.container():
        st.subheader("Input Delivery Parameters")
        
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Route Information")
                distance = st.slider("Distance (km)", 0.5, 20.0, 5.0, 0.1)
                traffic = st.select_slider("Traffic Level", 
                                          options=['Low', 'Medium', 'High'],
                                          value='Medium')
                weather = st.selectbox("Weather Condition", 
                                     ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy'])
                
            with col2:
                st.markdown("#### Delivery Details")
                vehicle = st.selectbox("Vehicle Type", ['Bike', 'Car', 'Scooter'])
                experience = st.slider("Courier Experience (years)", 0, 30, 2)
                prep_time = st.slider("Preparation Time (minutes)", 5, 30, 15)
                time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
            
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Estimated Delivery Time", 
                         f"{pred:.1f} minutes", 
                         delta=f"{diff:+.1f} vs average")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Delivery Time", 
                         f"{avg:.1f} minutes")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                efficiency = (1 - (pred/avg)) * 100 if pred < avg else 0
                if efficiency > 0:
                    st.metric("Efficiency Gain", 
                             f"{efficiency:.1f}% faster than average")
                else:
                    st.metric("Expected Delay", 
                             f"{-diff:.1f} minutes longer than average")
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
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.write("Top 5 Influencing Features:")
            st.dataframe(top_features[["Feature", "Coefficient"]].set_index("Feature")
                        .style.format("{:.2f}").background_gradient(cmap="RdBu", axis=0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=top_features, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
            ax.set_title("Top Features Affecting Delivery Time")
            ax.set_xlabel("Impact on Delivery Time (minutes)")
            ax.set_ylabel("")
            st.pyplot(fig)
        
        # Recommendations based on input
        st.markdown("---")
        st.subheader("Optimization Recommendations")
        
        if traffic == 'High':
            st.warning("🚦 High traffic detected! Consider using bikes/scooters which can navigate better in traffic.")
        
        if weather in ['Rainy', 'Snowy']:
            st.warning(f"☔ {weather} weather may cause delays. Consider adding time buffers for such conditions.")
        
        if distance > 15:
            st.info("📏 Long distance delivery. Using cars might be more efficient for distances over 15km.")
        
        if experience < 2:
            st.info("👶 New courier detected. More experienced couriers typically deliver faster.")
