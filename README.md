# Patient_dashboard
The Patient Health Record Dashboard is a data-driven web application built using Streamlit to help healthcare professionals analyze, monitor, and predict patient health conditions. This AI-powered dashboard provides real-time insights through interactive data visualizations and predictive analytics, improving patient care and decision-making.

Key Features
1. Patient Profiles 📋
The Patient Profiles section allows users to explore individual patient records, including key health metrics such as blood pressure, heart rate, and BMI. Users can select multiple patients for comparison, making it easier to analyze trends and identify potential health concerns. The system also supports search and filtering, allowing quick access to patient details.

2. Health Trends 📈
This section provides data visualization of key health metrics over time. Users can track trends in blood pressure, heart rate, and BMI through interactive Plotly charts. A rolling average feature smooths fluctuations, offering a clearer view of health patterns. These insights help healthcare professionals detect anomalies and assess overall patient well-being.

3. Predictive Analytics 🤖
The Predictive Analytics module employs a Random Forest Classifier to predict potential health risks. Users input key health parameters such as age, blood pressure, predicted risk score, and heart rate, and the model determines whether the patient is at high or low risk. The system also displays model accuracy and feature importance rankings, ensuring transparency and trust in AI-driven predictions.

4. Population Metrics 🌍
This feature provides aggregated health insights across a patient population. It calculates average blood pressure, glucose levels, and heart rate, offering a high-level view of health trends. The geospatial mapping system, built with Folium, visualizes patient distribution, helping hospitals and clinics understand demographic health patterns.

5. Performance Optimization 🚀
The dashboard uses Streamlit’s caching (@st.cache_data) to optimize performance by reducing redundant data loading. Machine learning models are stored in session state, preventing unnecessary retraining. Optimized data processing ensures faster response times, even with large datasets.

6. Settings & Customization ⚙️
The Settings section includes theme customization options (Light/Dark mode), enhancing user experience. Future updates will introduce more customization settings for UI and data presentation.

With its AI-driven insights, real-time data visualization, and performance optimizations, this dashboard empowers healthcare professionals to make informed decisions, predict risks, and improve patient outcomes. 🚀
