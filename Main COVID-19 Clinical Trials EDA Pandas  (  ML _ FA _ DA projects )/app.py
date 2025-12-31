import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
st.set_page_config(page_title="COVID-19 Clinical Trials Dashboard", layout="wide")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./COVID clinical trials.csv")
    return df

df = load_data()

# -----------------------
# Cleaning
# -----------------------
df.drop(columns=['Study Documents','Results First Posted'], inplace=True, errors='ignore')

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(f"Missing_{col}")

df['Enrollment'] = df['Enrollment'].fillna(df['Enrollment'].median())
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

df['Country'] = df['Locations'].apply(
    lambda x: x.split(',')[-1].strip() if pd.notna(x) else "Missing"
)

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.title("🔍 Filters")

selected_country = st.sidebar.multiselect(
    "Select Country",
    df["Country"].unique(),
    default=df["Country"].unique()[:5]
)

selected_status = st.sidebar.multiselect(
    "Select Trial Status",
    df["Status"].unique(),
    default=df["Status"].unique()
)

selected_phase = st.sidebar.multiselect(
    "Select Phase",
    df["Phases"].unique(),
    default=df["Phases"].unique()
)

top_n = st.sidebar.slider("Top N Countries", 5, 20, 10)

filtered_df = df[
    (df["Country"].isin(selected_country)) &
    (df["Status"].isin(selected_status)) &
    (df["Phases"].isin(selected_phase))
]

# -----------------------
# Title
# -----------------------
st.title("🧪 COVID-19 Clinical Trials – Interactive EDA Dashboard")
st.markdown("Explore global COVID-19 clinical trial data using interactive filters and charts.")

# -----------------------
# KPI Cards
# -----------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trials", filtered_df.shape[0])
col2.metric("Countries", filtered_df["Country"].nunique())
col3.metric("Median Enrollment", int(filtered_df["Enrollment"].median()))
col4.metric("Completed Trials", (filtered_df["Status"] == "Completed").sum())

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Time Analysis", "🌍 Country Analysis", "👥 Enrollment", "📥 Download"]
)

# -----------------------
# TAB 1: Overview
# -----------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20))

    if st.checkbox("Show Trial Status Distribution"):
        fig, ax = plt.subplots()
        filtered_df["Status"].value_counts().plot(kind="bar", ax=ax)
        plt.title("Trial Status Distribution")
        st.pyplot(fig)

    if st.checkbox("Show Phase Distribution"):
        fig, ax = plt.subplots()
        filtered_df["Phases"].value_counts().plot(kind="bar", ax=ax)
        plt.title("Phase Distribution")
        st.pyplot(fig)

# -----------------------
# TAB 2: Time Analysis
# -----------------------
with tab2:
    st.subheader("Trials Over Time")
    filtered_df["Start Month"] = filtered_df["Start Date"].dt.to_period("M")

    chart_type = st.radio("Select Chart Type", ["Line Chart", "Bar Chart"])

    fig, ax = plt.subplots()
    data = filtered_df["Start Month"].value_counts().sort_index()

    if chart_type == "Line Chart":
        data.plot(kind="line", ax=ax)
    else:
        data.plot(kind="bar", ax=ax)

    plt.title("Trials Started Over Time")
    st.pyplot(fig)

# -----------------------
# TAB 3: Country Analysis
# -----------------------
with tab3:
    st.subheader("Top Countries by Number of Trials")

    fig, ax = plt.subplots()
    filtered_df["Country"].value_counts().head(top_n).plot(kind="bar", ax=ax)
    plt.title(f"Top {top_n} Countries")
    st.pyplot(fig)

# -----------------------
# TAB 4: Enrollment
# -----------------------
with tab4:
    st.subheader("Enrollment Distribution")

    scale = st.radio("Select Scale", ["Normal", "Log Scale"])

    fig, ax = plt.subplots()
    ax.hist(filtered_df["Enrollment"], bins=100)

    if scale == "Log Scale":
        ax.set_yscale("log")

    plt.title("Enrollment Distribution")
    st.pyplot(fig)

# -----------------------
# TAB 5: Download
# -----------------------
with tab5:
    st.subheader("Download Cleaned Dataset")
    st.download_button(
        "Download CSV",
        filtered_df.to_csv(index=False),
        file_name="cleaned_covid_trials.csv",
        mime="text/csv"
    )
