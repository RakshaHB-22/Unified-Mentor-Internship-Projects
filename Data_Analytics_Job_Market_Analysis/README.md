# 📊 Data Analyst Job Market Analysis & Salary Prediction

An end-to-end data analytics and machine learning project that analyzes real-world Data Analyst job market trends and predicts salaries using Python and Streamlit. The application is containerized using Docker for consistent and scalable deployment.

---

## 📌 Project Overview

The COVID-19 pandemic significantly changed the global job market, especially in data-driven roles. This project analyzes over 2000 real-world Data Analyst job listings to uncover trends in salaries, locations, sectors, and skill requirements. A machine learning model is used to predict average salary based on selected job attributes, and the results are presented through an interactive Streamlit dashboard.

---

## 🎯 Objectives

- Analyze Data Analyst job market trends
- Identify high-paying job roles and locations
- Study the impact of skills like Python and Excel on salary
- Perform exploratory data analysis (EDA)
- Build a salary prediction model using Machine Learning
- Deploy an interactive dashboard using Streamlit

---

## 📂 Dataset Information

- **Source:** Glassdoor (via Kaggle)
- **Records:** 2253 job listings
- **Domain:** Job Market / Finance Analytics
- **Key Features:**
  - Job Title
  - Salary Estimate
  - Job Description
  - Company Rating
  - Location
  - Sector
  - Company Size
  - Link to download : https://drive.google.com/drive/folders/1Yl2rAwytABKjF1SOabxMe7civ0z1FBNI
---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Random Forest Regressor)
- **Web Framework:** Streamlit
- **Containerization:** Docker
- **Development Environment:** Jupyter Notebook, VS Code

---

## ⚙️ System Architecture

- **User → Dataset → Data Preprocessing → EDA → ML Model → Streamlit UI → User**


---

## 📊 Features

- Interactive dashboard for job market overview
- Salary distribution and trend analysis
- Sector-wise and location-wise salary insights
- Skill demand analysis (Python & Excel)
- Real-time salary prediction using ML
- Download options for dataset and notebook

---

## 🤖 Machine Learning Model

- **Model Used:** Random Forest Regressor
- **Input Features:** Company Rating, Python Skill, Excel Skill
- **Target Variable:** Average Salary
- **Evaluation Metrics:** MAE, R² Score

---

## 🖥️ Streamlit Application

The Streamlit app consists of:
- **Dashboard:** Job market summary
- **EDA Analysis:** Visual insights and trends
- **Salary Prediction:** ML-based prediction tool
- **Conclusion:** Key findings and download options

---

## 🐳 Dockerized Application

The application is containerized using Docker to ensure:
- Environment consistency
- Easy deployment
- Platform independence

### Docker Build & Run
```bash
docker build -t job-market-analysis-app .
docker run -p 8501:8501 job-market-analysis-app
```
Access the app at:
http://localhost:8501

---

Data_Analytics_Job_Market_Analysis/ <br>
│ <br>
├── app.py <br>
├── DataAnalyst.csv <br>
├── main.ipynb <br>
├── requirements.txt <br>
├── Dockerfile <br>
└── README.md <br>

---

## 👨‍💻 Author

- Pranuth Manjunath
- M.Tech | Data Analytics | Machine Learning
- GitHub: <href> https://github.com/PranuthHM </href>
