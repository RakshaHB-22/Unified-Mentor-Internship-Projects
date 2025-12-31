import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Customer Satisfaction Prediction",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./customer_support_tickets.csv")

df = load_data()

# ---------------------------------
# Train Model (NO joblib)
# ---------------------------------
@st.cache_resource
def train_model(df):

    df = df[df['Customer Satisfaction Rating'].notna()].copy()
    df['satisfied'] = (df['Customer Satisfaction Rating'] >= 4).astype(int)

    X = df[
        ['Ticket Description', 'Customer Age',
         'Ticket Type', 'Ticket Priority', 'Ticket Channel']
    ]
    y = df['satisfied']

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(
                stop_words='english',
                max_features=3000
            ), 'Ticket Description'),

            ('cat', OneHotEncoder(handle_unknown='ignore'),
             ['Ticket Type', 'Ticket Priority', 'Ticket Channel']),

            ('num', StandardScaler(), ['Customer Age'])
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model

model = train_model(df)

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "EDA Dashboard", "Model Performance", "Live Prediction", "Dataset Viewer"]
)

# ---------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------
if page == "Overview":

    st.title("📊 Customer Satisfaction Prediction System")

    st.markdown("""
    **Domain:** Data Science & Machine Learning  
    **Tech Stack:** Python, NLP, Scikit-Learn, Streamlit  

    This system predicts whether a customer will be **Satisfied or Not Satisfied**
    based on support ticket details using **NLP and Machine Learning**.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Tickets", len(df))
    col2.metric("Avg Satisfaction",
                round(df['Customer Satisfaction Rating'].mean(), 2))
    col3.metric("Satisfied Customers %",
                round((df['Customer Satisfaction Rating'] >= 4).mean() * 100, 2))

# ---------------------------------
# PAGE 2: EDA DASHBOARD
# ---------------------------------
elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satisfaction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Customer Satisfaction Rating', data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Ticket Priority Distribution")
        fig, ax = plt.subplots()
        df['Ticket Priority'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    st.subheader("Ticket Channel vs Satisfaction")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        x='Ticket Channel',
        y='Customer Satisfaction Rating',
        data=df,
        ax=ax
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Top 20 Words in Ticket Descriptions")
    words = " ".join(df['Ticket Description'].dropna()).lower().split()
    top_words = pd.Series(words).value_counts().head(20)
    st.bar_chart(top_words)

# ---------------------------------
# PAGE 3: MODEL PERFORMANCE
# ---------------------------------
elif page == "Model Performance":

    st.title("📊 Model Performance")

    df_eval = df.copy()
    df_eval['satisfied'] = (df_eval['Customer Satisfaction Rating'] >= 4).astype(int)

    X = df_eval[
        ['Ticket Description', 'Customer Age',
         'Ticket Type', 'Ticket Priority', 'Ticket Channel']
    ]
    y = df_eval['satisfied']

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y, y_prob)
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], '--')
    ax.legend()
    st.pyplot(fig)

# ---------------------------------
# PAGE 4: LIVE PREDICTION
# ---------------------------------
elif page == "Live Prediction":

    st.title("🤖 Live Customer Satisfaction Prediction")

    with st.form("prediction_form"):
        desc = st.text_area("Ticket Description")
        age = st.number_input("Customer Age", 18, 80, 30)
        ttype = st.selectbox(
            "Ticket Type", df['Ticket Type'].dropna().unique())
        priority = st.selectbox(
            "Ticket Priority", df['Ticket Priority'].dropna().unique())
        channel = st.selectbox(
            "Ticket Channel", df['Ticket Channel'].dropna().unique())
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            'Ticket Description': desc,
            'Customer Age': age,
            'Ticket Type': ttype,
            'Ticket Priority': priority,
            'Ticket Channel': channel
        }])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"😊 Customer is likely SATISFIED (Probability: {prob:.2f})")
        else:
            st.error(f"😞 Customer is likely NOT SATISFIED (Probability: {prob:.2f})")

# ---------------------------------
# PAGE 5: DATASET VIEWER
# ---------------------------------
elif page == "Dataset Viewer":

    st.title("📄 Dataset Viewer")

    st.dataframe(df)
    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        "customer_support_tickets.csv"
    )
