# app.py
import streamlit as st
import joblib, requests, re, numpy as np, pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from nltk.tokenize import sent_tokenize

import nltk

# Ensure punkt is available in Streamlit Cloud
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---- PATHS ----
MODEL_PATH = "models/quality_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
SVD_PATH = "models/tfidf_svd.pkl"
FEATURES_PATH = "models/feature_names.pkl"
TFIDF_MATRIX_PATH = "data/tfidf_matrix.npz"
FEATURES_CSV = "data/features_intermediate.csv"

# ---- LOAD ARTIFACTS (with safe error messages) ----
try:
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    svd = joblib.load(SVD_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    st.error("Error loading model artifacts. Make sure you ran training and files exist in /models.")
    st.exception(e)
    st.stop()

tfidf_matrix = None
df_features = None
try:
    tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
except Exception:
    tfidf_matrix = None

try:
    df_features = pd.read_csv(FEATURES_CSV)
except Exception:
    df_features = None

st.set_page_config(page_title="SEO Quality & Duplicate Detector", layout="wide")
st.title("SEO Content Quality & Duplicate Detector")

def clean_html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","nav","form"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r'\s+', ' ', text).strip()

def compute_features_from_text(text, raw_html=None):
    text = str(text)
    wc = len(text.split())
    sc = max(len(sent_tokenize(text)), 1)
    try:
        flesch = textstat.flesch_reading_ease(text)
    except:
        flesch = 50.0
    if raw_html is None:
        link_count = text.count("http")
        meta_count = 0
    else:
        soup = BeautifulSoup(raw_html, "html.parser")
        link_count = len(soup.find_all('a'))
        meta_count = len(soup.find_all('meta'))
    # Must match order of 'num_cols' used during training
    return [wc, link_count, meta_count, flesch]

# Sidebar input
st.sidebar.header("Input")
mode = st.sidebar.radio("Mode", ["URL", "Paste text", "Pick sample from dataset"])
text = ""
raw_html = None

if mode == "URL":
    url = st.sidebar.text_input("Enter URL (include https:// )")
    if st.sidebar.button("Fetch"):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                raw_html = r.text
                text = clean_html_to_text(raw_html)
            else:
                st.sidebar.error(f"Fetch failed {r.status_code}")
        except Exception as e:
            st.sidebar.error(str(e))

elif mode == "Paste text":
    text = st.sidebar.text_area("Paste article text here", height=200)

else:
    if df_features is not None:
        choose = st.sidebar.selectbox("Pick sample", df_features['url'].tolist()[:200])
        if choose:
            row = df_features[df_features['url']==choose].iloc[0]
            text = row['clean_text']
            raw_html = row.get('html_content', None)

# Main
if text and len(text.strip()) > 30:
    st.subheader("Content preview")
    st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

    with st.spinner("Computing features and prediction..."):
        feat_vals = compute_features_from_text(text, raw_html)
        v = vectorizer.transform([text])
        v_svd = svd.transform(v)
        # final feature array must be same shape as training -> numeric features then svd features
        X = np.hstack([np.array(feat_vals).reshape(1,-1), v_svd])
        try:
            pred_int = clf.predict(X)[0]
            label = le.inverse_transform([pred_int])[0]
        except Exception as e:
            st.error("Prediction failed — model and artifacts must match training versions.")
            st.exception(e)
            st.stop()

        st.metric("Predicted quality", label)
        st.write("Feature summary:", {
            "word_count": int(feat_vals[0]),
            "link_count": int(feat_vals[1]),
            "meta_count": int(feat_vals[2]),
            "flesch": round(float(feat_vals[3]), 2)
        })

        # duplicates (top 5) if available
        if tfidf_matrix is not None and df_features is not None:
            sims = cosine_similarity(v, tfidf_matrix).ravel()
            idxs = sims.argsort()[::-1][:5]
            st.subheader("Top similar pages in dataset (score)")
            for i in idxs:
                st.write(f"- {df_features.loc[i,'url']} — {sims[i]:.3f} — word_count={int(df_features.loc[i,'word_count'])}")
else:
    st.info("Provide URL, paste text or select sample from dataset.")

