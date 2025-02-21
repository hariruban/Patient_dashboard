import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure Page Settings
st.set_page_config(page_title="Patient Health Record Dashboard", layout="wide")
st.title("🩺 Patient Health Record Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Patient Profiles", "Health Trends", "Predictive Analytics", "Population Metrics", "Settings"])

# Load Data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data_file_path = "patient.csv"
data = load_data(data_file_path)

# Preprocess Data
@st.cache_data
def preprocess_data(df):
    if "Blood_Pressure" in df.columns:
        bp_split = df["Blood_Pressure"].str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors='coerce')
        df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors='coerce')
        df["Average_BP"] = (df["Systolic_BP"] + df["Diastolic_BP"]) / 2

    df["Potential_Health_Risk"] = df["Potential_Health_Risk"].astype(int)
    required_columns = ["Age", "Average_BP", "Predicted_Risk_Score", "Heart_Rate", "Potential_Health_Risk"]
    df = df.dropna(subset=required_columns)
    
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        df["Latitude"] = np.random.uniform(30.0, 50.0, len(df))
        df["Longitude"] = np.random.uniform(-120.0, -70.0, len(df))
    
    return df

data = preprocess_data(data)

if data is not None:
    st.sidebar.success("✅ Data Loaded Successfully!")

# Home Section
if section == "Home":
    st.header("🏠 Welcome to the Patient Health Dashboard")
    st.write("Monitor patient health, analyze trends, and predict risks with AI-driven insights.")
    st.image("patient.png", use_container_width=True)

# Patient Profiles
elif section == "Patient Profiles":
    st.header("📋 Patient Profiles")
    selected_patients = st.multiselect("Select Patients for Comparison", data["Patient_ID"].unique())
    st.dataframe(data[data["Patient_ID"].isin(selected_patients)] if selected_patients else data)

# Health Trends
elif section == "Health Trends":
    st.header("📈 Health Trends")
    metric = st.selectbox("Select Metric", ["Blood Pressure", "Heart Rate", "BMI"])
    metric_column = {"Blood Pressure": "Average_BP", "Heart Rate": "Heart_Rate", "BMI": "BMI"}[metric]
    data["Rolling_Avg"] = data[metric_column].rolling(window=5).mean()
    fig = px.line(data, x="Patient_ID", y=[metric_column, "Rolling_Avg"], title=f"{metric} Trends")
    st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics
elif section == "Predictive Analytics":
    st.header("🤖 Predictive Analytics")
    X = data[["Age", "Average_BP", "Predicted_Risk_Score", "Heart_Rate"]]
    y = data["Potential_Health_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if "model" not in st.session_state:
        st.session_state.model = RandomForestClassifier(random_state=42)
        st.session_state.model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, st.session_state.model.predict(X_test))
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    user_input_df = pd.DataFrame([[st.number_input(label, min_value=0.0) for label in ["Age", "Average_BP", "Predicted_Risk_Score", "Heart_Rate"]]], columns=X.columns)
    if st.button("Predict Health Risk"):
        prediction = st.session_state.model.predict(user_input_df)[0]
        st.success("High Risk" if prediction else "Low Risk")

    st.write("Feature Importances:")
    st.bar_chart(pd.DataFrame({"Feature": X.columns, "Importance": st.session_state.model.feature_importances_}).set_index("Feature"))

# Population Metrics
elif section == "Population Metrics":
    st.header("🌍 Population Metrics")
    @st.cache_data
    def compute_metrics(df):
        return {
            "avg_bp": df["Average_BP"].mean(),
            "avg_glucose": df["Predicted_Risk_Score"].mean(),
            "avg_hr": df["Heart_Rate"].mean()
        }
    
    metrics = compute_metrics(data)
    st.metric("Avg Blood Pressure", f"{metrics['avg_bp']:.2f} mmHg")
    st.metric("Avg Glucose Level", f"{metrics['avg_glucose']:.2f} mg/dL")
    st.metric("Avg Heart Rate", f"{metrics['avg_hr']:.2f} bpm")
    
    st.subheader("Geospatial Patient Distribution")
    sampled_data = data.sample(n=min(500, len(data)), random_state=42)
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=5)
    for _, row in sampled_data.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]], popup=f"Patient {row['Patient_ID']}").add_to(m)
    folium_static(m)

# Settings Section
elif section == "Settings":
    st.header("⚙️ Settings")
    theme_choice = st.radio("Choose Theme", ["Light", "Dark"])
    st.write("🚀 Future updates will include custom theme settings.")
