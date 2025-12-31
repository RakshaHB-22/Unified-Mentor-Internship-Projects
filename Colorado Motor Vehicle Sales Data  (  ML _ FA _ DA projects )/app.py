import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Motor Vehicle Sales Dashboard", layout="wide")

st.title("🚗 Colorado Motor Vehicle Sales Analysis")

# Load data
df = pd.read_csv("colorado_motor_vehicle_sales.csv")

st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Select Section",
    ["Overview", "EDA", "County Analysis", "Forecasting"]
)

if option == "Overview":
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.metric("Total Sales", f"${df.sales.sum():,.0f}")
    st.metric("Average Sales", f"${df.sales.mean():,.0f}")

elif option == "EDA":
    st.subheader("Sales Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['quarter'], y=df['sales'], ax=ax)
    st.pyplot(fig)

elif option == "County Analysis":
    year = st.selectbox("Select Year", sorted(df.year.unique()))
    quarter = st.selectbox("Select Quarter", [1,2,3,4])

    temp = df[(df.year==year) & (df.quarter==quarter)]
    county_sales = temp.groupby('county')['sales'].sum()

    fig, ax = plt.subplots(figsize=(10,4))
    county_sales.sort_values(ascending=False).plot(kind='bar', ax=ax)
    st.pyplot(fig)

elif option == "Forecasting":
    df['month'] = df['quarter'].map({1:1,2:4,3:7,4:10})
    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
    ts = df.groupby('date')['sales'].sum().asfreq('Q')

    model = ARIMA(ts, order=(1,1,1))
    fit = model.fit()
    forecast = fit.forecast(steps=8)

    fig, ax = plt.subplots()
    ax.plot(ts, label="Actual")
    ax.plot(forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)
