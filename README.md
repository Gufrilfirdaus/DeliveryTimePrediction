This project aims to predict delivery duration based on historical food delivery data using regression modeling. The goal is to provide operational insights and support smarter scheduling in logistics by identifying key delay factors.

📌 Project Objectives
Identify key factors that affect delivery time (e.g., distance, weather, courier experience).

Build a regression model to accurately predict delivery duration.

Recommend data-driven actions to improve logistics performance.

📊 Dataset
1,000 records

9 features: Distance, Weather, Traffic_Level, Time_of_Day, Vehicle_Type, Preparation_Time, Courier_Experience, etc.

Cleaned and encoded using:

Median/Mode for missing values

One-Hot Encoding for nominal features

Label Encoding for ordinal features

🔍 Exploratory Data Analysis (EDA)
Key findings:

Deliveries over long distances in the morning had the highest average duration (79.4 mins).

Snowy weather and junior couriers significantly increased delivery time.

Distance showed strong positive correlation with delivery time (r = 0.78).

Preparation time peaked during long deliveries in the morning (avg. 17.3 mins).

🤖 Modeling
Model used: Linear Regression

Result: Strong performance with accurate delivery time estimation

Most impactful feature: Distance

💡 Recommendations
Assign senior couriers to long-distance and morning deliveries.

Avoid long-distance orders during peak hours, especially in bad weather.

Use cars instead of bikes in snowy conditions for faster delivery.

🛠️ Tools & Libraries
Python

Pandas

Seaborn

Scikit-learn
