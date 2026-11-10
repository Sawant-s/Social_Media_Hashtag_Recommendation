"""
app.py
------
Streamlit web application for the
"Social Media Hashtag Recommendation System".

What this app does:
1. Takes a social media caption typed by the user.
2. Cleans/preprocesses the text (same way as during training).
3. Converts it into numbers using the saved TF-IDF vectorizer.
4. Predicts the CATEGORY of the post using the trained ML model.
5. Recommends hashtags that belong to that category
   (built from the same dataset the model was trained on).

Run with:
    streamlit run app.py
"""

import re
import string
import warnings

import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PAGE CONFIG - must be the first Streamlit command
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Hashtag Recommendation System",
    page_icon="#️⃣",
    layout="centered",
)

# ---------------------------------------------------------------------
# CUSTOM CSS - gives the app a clean, modern, social-media inspired look
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #833ab4, #fd1d1d, #fcb045);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #6b6b6b;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .category-card {
        background: linear-gradient(135deg, #833ab4, #fd1d1d);
        color: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 15px 0px;
    }
    .hashtag-chip {
        display: inline-block;
        background-color: #f0f2ff;
        color: #833ab4;
        border: 1px solid #d9d4ff;
        padding: 7px 14px;
        margin: 5px 6px 5px 0px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .info-box {
        background-color: #f7f7f9;
        border-left: 4px solid #833ab4;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #444;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# TEXT PREPROCESSING - must exactly match the cleaning used in training
# ---------------------------------------------------------------------
def clean_text(text):
    """Lowercase, remove emojis/links/punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.encode("ascii", "ignore").decode("ascii")   # remove emojis
    text = re.sub(r"http\S+|www\S+", " ", text)               # remove links
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------
# LOAD MODEL, VECTORIZER AND DATASET (cached so it only loads once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_hashtag_map():
    """
    Build a dictionary: {category: [list of unique hashtags]}
    by reading the same dataset the model was trained on.
    This is how new hashtags are recommended for a predicted category.
    """
    df = pd.read_csv("dataset.csv")
    hashtag_map = {}

    for category in df["category"].unique():
        tags_for_category = df.loc[df["category"] == category, "hashtags"]

        seen = []
        for row in tags_for_category:
            for tag in row.split():
                if tag not in seen:
                    seen.append(tag)

        hashtag_map[category] = seen

    return hashtag_map


model, vectorizer = load_model_and_vectorizer()
hashtag_map = load_hashtag_map()


# ---------------------------------------------------------------------
# SESSION STATE - keeps the caption box value across button clicks
# ---------------------------------------------------------------------
if "caption_text" not in st.session_state:
    st.session_state.caption_text = ""
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False


def reset_app():
    st.session_state.caption_text = ""
    st.session_state.result_ready = False


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown('<div class="main-title">📱 Social Media Hashtag Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered hashtag suggestions based on your caption\'s category</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------------
caption = st.text_area(
    "Enter your social media caption",
    value=st.session_state.caption_text,
    height=120,
    placeholder="e.g. Exploring the beautiful mountains with my friends.",
    key="caption_input",
)

col1, col2 = st.columns([1, 1])
with col1:
    recommend_clicked = st.button("🔮 Recommend Hashtags", use_container_width=True, type="primary")
with col2:
    reset_clicked = st.button("🔄 Clear / Reset", use_container_width=True, on_click=reset_app)

# ---------------------------------------------------------------------
# WHEN "RECOMMEND HASHTAGS" IS CLICKED
# ---------------------------------------------------------------------
if recommend_clicked:
    st.session_state.caption_text = caption

    # Handle empty input
    if caption is None or caption.strip() == "":
        st.warning("⚠️ Please enter a caption before requesting hashtags.")
        st.session_state.result_ready = False
    else:
        # 1. Preprocess
        cleaned = clean_text(caption)

        # 2. Convert to TF-IDF vector
        vector = vectorizer.transform([cleaned])

        # 3. Predict category
        predicted_category = model.predict(vector)[0]

        # 4. Get prediction confidence for a simple visualization
        proba = model.predict_proba(vector)[0]
        prob_df = pd.DataFrame({
            "Category": model.classes_,
            "Confidence": proba
        }).sort_values("Confidence", ascending=False).head(5)

        # 5. Recommend hashtags (5 to 10) for the predicted category
        all_tags = hashtag_map.get(predicted_category, [])
        recommended_tags = all_tags[:10] if len(all_tags) >= 5 else all_tags

        # Save to session state so results survive reruns
        st.session_state.result_ready = True
        st.session_state.predicted_category = predicted_category
        st.session_state.recommended_tags = recommended_tags
        st.session_state.prob_df = prob_df

# ---------------------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------------------
if st.session_state.result_ready:
    st.markdown(
        f'<div class="category-card">🏷️ Predicted Category: {st.session_state.predicted_category}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Recommended Hashtags")

    tags = st.session_state.recommended_tags
    chips_html = "".join([f'<span class="hashtag-chip">{tag}</span>' for tag in tags])
    st.markdown(chips_html, unsafe_allow_html=True)

    # Copy-friendly version (Streamlit's code blocks have a built-in copy button)
    st.markdown("###### Copy Hashtags")
    st.code(" ".join(tags), language=None)

    # Simple statistics
    st.markdown("#### 📊 Stats & Confidence")
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Hashtags Recommended", len(tags))
    with stat_col2:
        top_confidence = st.session_state.prob_df.iloc[0]["Confidence"] * 100
        st.metric("Model Confidence", f"{top_confidence:.1f}%")

    st.bar_chart(st.session_state.prob_df.set_index("Category")["Confidence"])

    st.markdown(
        '<div class="info-box">ℹ️ These hashtags were recommended based on the detected post category.</div>',
        unsafe_allow_html=True,
    )

