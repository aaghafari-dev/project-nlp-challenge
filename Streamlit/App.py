""" Project: Fake News Classifier – NLP Challenge (Week 30)""" 
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import graphviz  # <-- added for flowchart

# Set page config
st.set_page_config(page_title="Fake News Classifier", layout="wide")

# Custom CSS for sidebar (light blue) and metric cards
st.markdown(
    """
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #ADD8E6;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf5 100%);
        border-radius: 12px;
        padding: 20px 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px 0;
    }
    .metric-card h3 {
        margin: 0;
        color: #2c3e50;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .metric-card p {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0 0 0;
        color: #1a5276;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load pipeline and test results
@st.cache_resource
def load_artifacts():
    with open("../notebook/final_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("../notebook/test_results.pkl", "rb") as f:
        results = pickle.load(f)
    return pipeline, results

pipeline, results = load_artifacts()

# Unpack test results
test_data = results['test_data']
predictions = results['predictions']
probabilities = results['probabilities']
conf_matrix = results['confusion_matrix']
class_report = results['classification_report']
feature_names = results.get('feature_names', None)
coefs = results.get('coefficients', None)

# Load training data for EDA
@st.cache_data
def load_training_data():
    data_path = os.path.join("..", "dataset", "data.csv")
    if not os.path.exists(data_path):
        data_path = "data.csv"
    return pd.read_csv(data_path)

train_df = load_training_data()

# Sidebar 
page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Live Prediction",
        "Confusion Matrix",
        "Feature Importance",
        "Conclusion",
        "Acknowledgements"
    ]
)

# ============================
# OVERVIEW (includes contribution, flowchart, EDA)
# ============================
if page == "Overview":
    st.header("Model Performance on Test Set")

    # ----- My Contribution -----
    st.subheader("My Contribution")
    st.markdown("""
    - Designed and implemented the end‑to‑end machine learning pipeline.
    - Performed exploratory data analysis (EDA) to understand data distribution and guide preprocessing.
    - Engineered three feature sets (title‑only, text‑only, combined) and evaluated multiple classifiers via cross‑validation.
    - Optimised hyperparameters using a systematic grid search with 5‑fold stratified cross‑validation.
        """)

    # ----- Flowchart using Graphviz -----
    st.subheader("Project Workflow")
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", splines="ortho")
    dot.attr("node", shape="box", style="rounded, filled", fillcolor="#ADD8E6", fontname="Helvetica")
    dot.node("A", "Data Loading")
    dot.node("B", "EDA")
    dot.node("C", "Text Preprocessing\n(POS-aware lemmatisation)")
    dot.node("D", "Feature Engineering\n(title, text, combined)")
    dot.node("E", "Model Selection\n(Grid search + CV)")
    dot.node("F", "Evaluation\n(Test set metrics)")
    dot.node("G", "Deployment\n(Streamlit App)")

    dot.edge("A", "B")
    dot.edge("B", "C")
    dot.edge("C", "D")
    dot.edge("D", "E")
    dot.edge("E", "F")
    dot.edge("F", "G")

    st.graphviz_chart(dot)

    # ----- Metric cards -----
    st.subheader("Performance Metrics")
    st.markdown("""
    <div style="display: flex; justify-content: space-around; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
        <div class="metric-card">
            <h3>Accuracy</h3>
            <p>{acc:.4f}</p>
        </div>
        <div class="metric-card">
            <h3>F1 (weighted)</h3>
            <p>{f1:.4f}</p>
        </div>
        <div class="metric-card">
            <h3>Test samples</h3>
            <p>{n}</p>
        </div>
    </div>
    """.format(
        acc=class_report['accuracy'],
        f1=class_report['weighted avg']['f1-score'],
        n=len(test_data)
    ), unsafe_allow_html=True)

    # Classification report
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(class_report).transpose().round(4))

    # ----- EDA section (embedded) -----
    st.markdown("---")
    st.header("Exploratory Data Analysis (Training Data)")

    # Label distribution
    st.subheader("Label Distribution")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    sns.countplot(x='label', data=train_df, ax=ax1)
    ax1.set_title("Distribution of Labels")
    st.pyplot(fig1)

    # Subject distribution
    st.subheader("Subject Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    train_df['subject'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title("News Subjects")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # Text length analysis
    st.subheader("Text Length by Label")
    train_df['title_len'] = train_df['title'].apply(len)
    train_df['text_len'] = train_df['text'].apply(len)
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x='label', y='title_len', data=train_df, ax=axes[0])
    axes[0].set_title('Title Length by Label')
    sns.boxplot(x='label', y='text_len', data=train_df, ax=axes[1])
    axes[1].set_title('Text Length by Label')
    st.pyplot(fig3)

    # Summary
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    col1.metric("Total samples", train_df.shape[0])
    col2.metric("Missing values", train_df.isnull().sum().sum())
    st.write("Class counts:")
    st.write(train_df['label'].value_counts().to_frame().T)

# ============================
# LIVE PREDICTION
# ============================
elif page == "Live Prediction":
    st.header("Live Prediction on Test Samples")
    st.write("Select a test sample below to see the model’s output – all results are pre‑computed from the notebook.")

    sample_options = [f"{i+1}: {row['title'][:60]}..." for i, (_, row) in enumerate(test_data.iterrows())]
    selected = st.selectbox("Choose a test headline", sample_options)
    idx = int(selected.split(":")[0]) - 1

    row = test_data.iloc[idx]
    true_label = row['label']
    pred_label = predictions[idx]
    prob_real = probabilities[idx]
    prob_fake = 1 - prob_real

    st.subheader("Sample Text")
    st.write(f"**Title:** {row['title']}")
    st.write(f"**Text:** {row['text'][:500]}...")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("True Label", "Real" if true_label == 1 else "Fake")
    with col2:
        st.metric("Predicted Label", "Real" if pred_label == 1 else "Fake",
                  delta="Correct" if pred_label == true_label else "Incorrect")

    st.subheader("Confidence Scores")
    st.progress(prob_real, text=f"Real: {prob_real:.2%}")
    st.progress(prob_fake, text=f"Fake: {prob_fake:.2%}")
    st.caption(f"Probability of being Real: {prob_real:.4f}")

# ============================
# CONFUSION MATRIX
# ============================
elif page == "Confusion Matrix":
    st.header("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# ============================
# FEATURE IMPORTANCE
# ============================
elif page == "Feature Importance":
    st.header("Top Predictive Features")
    if feature_names is not None and coefs is not None:
        top_n = 20
        pos_idx = np.argsort(coefs)[-top_n:][::-1]
        neg_idx = np.argsort(coefs)[:top_n]
        top_pos = [(feature_names[i], coefs[i]) for i in pos_idx]
        top_neg = [(feature_names[i], coefs[i]) for i in neg_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Real‑indicative")
            for word, coef in top_pos:
                st.write(f"{word}: {coef:.4f}")
        with col2:
            st.subheader("Fake‑indicative")
            for word, coef in top_neg:
                st.write(f"{word}: {coef:.4f}")
    else:
        st.info("Feature importance not available for this model (non‑linear).")

# ============================
# CONCLUSION
# ============================
elif page == "Conclusion":
    st.header("Conclusion")
    st.markdown("""
    **Project Summary**

    - The binary classifier was trained on a dataset of ~40,000 news articles (70% training, 30% test).
    - Text preprocessing included POS‑aware lemmatisation and stopword removal.
    - The best configuration (selected via 5‑fold cross‑validation) was:
        - **Feature set**: Combined (title + text)
        - **Classifier**: Logistic Regression (C=10, L1 penalty)
    - On the held‑out test set, the model achieved:
        - **Accuracy**: {acc:.4f}
        - **Weighted F1‑score**: {f1:.4f}
    - The confusion matrix shows almost perfect classification, indicating the model is very effective on in‑distribution data.
       
    """.format(
        acc=class_report['accuracy'],
        f1=class_report['weighted avg']['f1-score']
    ))

# ============================
# ACKNOWLEDGEMENTS
# ============================
else:
    st.header("Acknowledgements")
    st.markdown("""
    **Thanks, Cristina and Rami, for your support**  
    **Thank you for your attention!**
    """)