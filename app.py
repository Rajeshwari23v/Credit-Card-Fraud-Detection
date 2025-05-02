import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Custom styling for a compact and interactive layout
st.markdown("""
    <style>
        body {
            background-color: #e8f5e9; /* Light Green Background */
        }
        .main {
            background: linear-gradient(to right, #a5d6a7, #81c784);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            width: 90%;
        }
        .stButton>button {
            background-color: #388e3c;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
        }
        h1, h2 {
            color: #1b5e20;
            font-size: 1.5rem;
        }
        .stSidebar {
            width: 260px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title
st.title("💳 Credit Card Fraud Detection")

# Brief introduction
st.markdown("""
This app detects fraudulent credit card transactions using machine learning. 
You can choose between **Random Forest** and **Logistic Regression** models, 
apply **SMOTE** (oversampling), and evaluate performance using **Precision**, **Recall**, and **F1-score**.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop('Time', axis=1)
    return df

df = load_data()

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.selectbox("Select Classifier", ["Random Forest", "Logistic Regression"])
sample_size = st.sidebar.slider("Training Sample Size", min_value=1000, max_value=30000, step=1000, value=10000)
apply_smote = st.sidebar.checkbox("Apply SMOTE (Oversampling)", value=True)
run = st.sidebar.button("🚀 Run Detection")

# Show data preview
with st.expander("📄 Preview Data"):
    st.write(df.head())

# Visualize the fraud and genuine distribution
if st.checkbox("📊 Show Class Distribution"):
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size for compactness
    sns.countplot(data=df, x='Class', palette='Set2', ax=ax)
    ax.set_title("Fraud vs Genuine Distribution")
    st.pyplot(fig)

# Run Model
if run:
    st.subheader("🔍 Model Training & Evaluation")

    # Sample and separate data
    df_sample = pd.concat([df[df['Class'] == 0].sample(sample_size), df[df['Class'] == 1]])
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']

    # Apply SMOTE if selected
    if apply_smote:
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train the selected model
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display classification report
    with st.expander("📊 Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix visualization
    with st.expander("📉 Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))  # Compact size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    # Display the metrics
    st.markdown("### 📈 Evaluation Metrics Summary")
    st.markdown(f"- **Precision:** `{precision:.4f}`")
    st.markdown(f"- **Recall:** `{recall:.4f}`")
    st.markdown(f"- **F1-score:** `{f1:.4f}`")

    st.success("✅ Model completed! Check your metrics above.")

st.markdown("</div>", unsafe_allow_html=True)
