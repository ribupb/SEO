import streamlit as st
import joblib, requests, re, numpy as np, pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from nltk.tokenize import sent_tokenize

# ------------------ Load Artifacts ------------------
MODEL_PATH = "models/quality_model22.pkl"
ENCODER_PATH = "models/label_encoder22.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
SVD_PATH = "models/tfidf_svd.pkl"
FEATURES_PATH = "models/feature_names.pkl"

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
svd = joblib.load(SVD_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Optional: duplicate detection files
try:
    tfidf_matrix = load_npz("data/tfidf_matrix.npz")
    df_features = pd.read_csv("data/features_intermediate.csv")
except Exception:
    tfidf_matrix, df_features = None, None

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="SEO Quality & Duplicate Detector", layout="wide")
st.title("SEO Content Quality & Duplicate Detector")

def clean_html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript","header","footer","nav","form"]):
        t.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r'\s+', ' ', text).strip()

def compute_features_from_text(text, raw_html=None):
    wc = len(text.split())
    try: flesch = textstat.flesch_reading_ease(text)
    except: flesch = 50.0
    if raw_html is None:
        link_count = text.count("http")
        meta_count = 0
    else:
        soup = BeautifulSoup(raw_html, "html.parser")
        link_count = len(soup.find_all("a"))
        meta_count = len(soup.find_all("meta"))

    # Order must match feature_names.pkl (from training)
    return [wc, link_count, meta_count, flesch]


st.sidebar.header("Input")
mode = st.sidebar.radio("Mode", ["URL", "Paste text"])

text = ""
raw_html = None

if mode == "URL":
    url = st.sidebar.text_input("Enter URL (with https://)")
    if st.sidebar.button("Fetch"):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                raw_html = r.text
                text = clean_html_to_text(raw_html)
            else:
                st.sidebar.error("Failed: HTTP " + str(r.status_code))
        except Exception as e:
            st.sidebar.error(str(e))

else:  # Paste text mode
    text = st.sidebar.text_area("Paste content", height=200)

if len(text) > 30:
    st.subheader("Content Preview")
    st.write(text[:500] + "...")

    with st.spinner("Predicting..."):

        feat_vals = compute_features_from_text(text, raw_html)
        v = vectorizer.transform([text])
        v_svd = svd.transform(v)
        X = np.hstack([np.array(feat_vals).reshape(1,-1), v_svd])

        pred_class = clf.predict(X)[0]
        label = le.inverse_transform([pred_class])[0]

        st.metric("Predicted Quality", label)

        st.write("Extracted features:", {
            "word_count": int(feat_vals[0]),
            "link_count": int(feat_vals[1]),
            "meta_count": int(feat_vals[2]),
            "flesch": round(float(feat_vals[3]),2)
        })

        if tfidf_matrix is not None:
            sims = cosine_similarity(v, tfidf_matrix).ravel()
            idx = sims.argsort()[::-1][:5]
            st.subheader("Similar Pages in Dataset")
            for i in idx:
                st.write(f"{df_features.loc[i, 'url']} — score: {sims[i]:.3f}")

else:
    st.info("Enter URL or Paste Content to start 🔍")
