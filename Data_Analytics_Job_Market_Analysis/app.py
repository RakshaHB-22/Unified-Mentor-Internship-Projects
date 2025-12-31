import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Data Analyst Job Market", layout="wide")

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 Data Analyst Job Market Analysis & Salary Prediction")
st.markdown("**Domain:** Finance Analyst | **Tools:** Python, ML, Streamlit")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("DataAnalyst.csv")

    # Drop unwanted columns
    data.drop(['Unnamed: 0', 'Founded', 'Competitors'], axis=1, inplace=True)

    # Rename columns
    data.rename(columns={
        "Job Title": "job_title",
        "Salary Estimate": "salary_estimate",
        "Job Description": "job_description",
        "Company Name": "company_name",
        "Location": "location",
        "Headquarters": "headquarters",
        "Size": "size",
        "Type of ownership": "type_of_ownership",
        "Industry": "industry",
        "Sector": "sector",
        "Revenue": "revenue",
        "Easy Apply": "easy_apply"
    }, inplace=True)

    # Salary extraction
    salary = data['salary_estimate'].str.extract(r'\$(\d+)K-\$(\d+)K')
    data['MinSalary'] = pd.to_numeric(salary[0], errors='coerce')
    data['MaxSalary'] = pd.to_numeric(salary[1], errors='coerce')
    data['average_salary'] = (data['MinSalary'] + data['MaxSalary']) / 2

    data.dropna(subset=['average_salary'], inplace=True)

    # Skill extraction
    data['Python'] = data['job_description'].str.contains('python', case=False).astype(int)
    data['Excel'] = data['job_description'].str.contains('excel', case=False).astype(int)

    return data

data = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("📌 Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Dashboard", "EDA Analysis", "Salary Prediction", "Conclusion"]
)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if section == "Dashboard":
    st.subheader("📈 Job Market Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Jobs", len(data))
    col2.metric("Average Salary", f"${data['average_salary'].mean():.0f}K")
    col3.metric("Top Job Title", data['job_title'].value_counts().idxmax())

    st.markdown("---")

    st.subheader("Top 10 Job Titles")
    fig, ax = plt.subplots()
    sns.barplot(
        y=data['job_title'].value_counts().head(10).index,
        x=data['job_title'].value_counts().head(10).values,
        ax=ax
    )
    st.pyplot(fig)

# -------------------------------------------------
# EDA
# -------------------------------------------------
elif section == "EDA Analysis":
    st.subheader("📊 Exploratory Data Analysis")

    # ---------------- Salary Distribution ----------------
    st.markdown("### Salary Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxenplot(x=data['average_salary'], ax=ax)
    ax.set_title("Distribution of Average Salary")
    ax.set_xlabel("Average Salary (K USD)")
    ax.set_ylabel("Density")
    st.pyplot(fig)

    # ---------------- Salary by Sector ----------------
    st.markdown("### Salary by Sector")
    sector_salary = (
        data.groupby('sector')['average_salary']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        y=sector_salary.index,
        x=sector_salary.values,
        ax=ax
    )
    ax.set_title("Top 10 Sectors by Average Salary")
    ax.set_xlabel("Average Salary (K USD)")
    ax.set_ylabel("Sector")
    st.pyplot(fig)

    # ---------------- Salary by Location ----------------
    st.markdown("### Top Paying Locations")
    location_salary = (
        data.groupby('location')['average_salary']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        y=location_salary.index,
        x=location_salary.values,
        ax=ax
    )
    ax.set_title("Top 10 Locations by Average Salary")
    ax.set_xlabel("Average Salary (K USD)")
    ax.set_ylabel("Location")
    st.pyplot(fig)

# -------------------------------------------------
# SALARY PREDICTION
# -------------------------------------------------
elif section == "Salary Prediction":
    st.subheader("🤖 Salary Prediction Using Machine Learning")

    # Features & Target
    X = data[['Rating', 'Python', 'Excel']]
    y = data['average_salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User Inputs
    rating = st.slider("Company Rating", 1.0, 5.0, 3.5, 0.1)
    python_skill = st.checkbox("Python Required")
    excel_skill = st.checkbox("Excel Required")

    python_val = 1 if python_skill else 0
    excel_val = 1 if excel_skill else 0

    if st.button("Predict Salary"):
        prediction = model.predict([[rating, python_val, excel_val]])
        st.success(f"💰 Predicted Average Salary: ₹{prediction[0] * 1000 * 83:,.2f} per year")


# -------------------------------------------------
# CONCLUSION
# -------------------------------------------------
elif section == "Conclusion":
    st.subheader("📌 Project Conclusion")

    st.markdown("""
    **Key Insights:**
    - Data Analyst roles are in high demand across industries
    - Newark locations offer the highest salary packages
    - Python and Excel skills significantly impact salary levels
    - Sector and location influence salary more than company rating

    **Outcome:**
    This project delivers an end-to-end job market analysis and
    salary prediction system that supports data-driven decision making
    for job seekers and recruiters.
    """)

    st.markdown("---")
    st.subheader("📥 Download Project Files")

    col1, col2 = st.columns(2)

    # Download Dataset
    with col1:
        with open("DataAnalyst.csv", "rb") as file:
            st.download_button(
                label="⬇️ Download Dataset (CSV)",
                data=file,
                file_name="DataAnalyst.csv",
                mime="text/csv"
            )

    # Download Jupyter Notebook
    with col2:
        with open("main.ipynb", "rb") as file:
            st.download_button(
                label="⬇️ Download Jupyter Notebook",
                data=file,
                file_name="main.ipynb",
                mime="application/octet-stream"
            )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Developed as an Academic Main Project | Data Analyst & Finance Domain")
