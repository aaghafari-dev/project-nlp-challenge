"""
Streamlit App for Project: Fake News Classifier – NLP Challenge (Week 30)
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------
# 1. Download NLTK resources (run once)
# -----------------------------------------------------------------------------
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)

# -----------------------------------------------------------------------------
# 2. Load the pre-trained pipeline (cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the serialised pipeline from disk.
    Returns:
        sklearn.pipeline.Pipeline: the trained model pipeline.
    """
    with open('final_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

# -----------------------------------------------------------------------------
# 3. Text preprocessing function (must match training)
# -----------------------------------------------------------------------------
# Constants for cleaning (defined once for efficiency)
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
# Map NLTK POS tags to WordNet tags (first character)
WORDNET_TAG_MAP = {'J': 'a', 'V': 'v', 'R': 'r'}  # default 'n' for noun

def clean_text(text: str) -> str:
    """
    Clean and lemmatize input text using POS-aware lemmatization.
    This function must be identical to the one used during training.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, lemmatized text.
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove non-alphabetic characters from each token (keep only letters)
    tokens = [re.sub(r'[^a-zA-Z]', '', t) for t in tokens]

    # Remove stopwords and empty tokens
    tokens = [t for t in tokens if t and t not in STOP_WORDS]

    # POS tagging
    tagged = pos_tag(tokens)

    # Lemmatize with POS mapping
    lemmatized = [
        LEMMATIZER.lemmatize(word, WORDNET_TAG_MAP.get(tag[0], 'n'))
        for word, tag in tagged
    ]

    return ' '.join(lemmatized)

# -----------------------------------------------------------------------------
# 4. Streamlit UI Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon=":newspaper:",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 5. Sidebar – Theme Customisation
# -----------------------------------------------------------------------------
st.sidebar.header("Theme Customisation")

# Colour pickers
bg_color = st.sidebar.color_picker("Background Colour", "#f0f2f6")
sidebar_bg = st.sidebar.color_picker("Sidebar Colour", "#e8eaf0")
button_bg = st.sidebar.color_picker("Button Background", "#4CAF50")
button_text = st.sidebar.color_picker("Button Text Colour", "#ffffff")

# Font settings
font_family = st.sidebar.selectbox(
    "Font Family",
    ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"]
)
font_size = st.sidebar.slider("Font Size (px)", 12, 24, 16)

# -----------------------------------------------------------------------------
# 6. Inject Custom CSS
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: {bg_color};
        }}
        .sidebar .sidebar-content {{
            background: {sidebar_bg};
        }}
        .stButton>button {{
            color: {button_text};
            background: {button_bg};
            font-size: {font_size}px;
            font-family: {font_family};
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }}
        .stMarkdown, .stTextInput, .stSelectbox {{
            font-family: {font_family};
            font-size: {font_size}px;
        }}
        h1, h2, h3 {{
            font-family: {font_family};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 7. Main Interface
# -----------------------------------------------------------------------------
st.title("Fake News Classifier")
st.markdown(
    "Enter a news headline and/or full text to predict if it is **Real** (1) or **Fake** (0)."
)

# Input columns
col1, col2 = st.columns(2)
with col1:
    headline = st.text_area("Headline", height=100)
with col2:
    full_text = st.text_area("Full Text", height=200)

# Prediction button
if st.button("Predict"):
    if headline or full_text:
        # Combine inputs
        combined = headline + " " + full_text

        # Clean the combined text using the same function as training
        cleaned = clean_text(combined)

        # Predict
        pred = pipeline.predict([cleaned])[0]

        # Get probability if the classifier supports it
        if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
            prob = pipeline.predict_proba([cleaned])[0][1]
        else:
            prob = None

        # Display result
        result = "Real" if pred == 1 else "Fake"
        st.subheader(f"Prediction: {result}")
        if prob is not None:
            st.write(f"Confidence (probability of being Real): {prob:.2f}")
    else:
        st.warning("Please enter at least a headline or some text.")