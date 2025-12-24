import os
import io
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Animal Classifier AI",
    page_icon="🐾",
    layout="wide",
)

# =========================
# CONSTANTS
# =========================
MODEL_PATH = "animal_model.keras"
IMG_SIZE = (256, 256)

class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer',
    'Dog', 'Dolphin', 'Elephant', 'Giraffe',
    'Horse', 'Kangaroo', 'Lion', 'Panda',
    'Tiger', 'Zebra'
]

labels_en = {
    "Bear": "🐻 Bear",
    "Bird": "🐦 Bird",
    "Cat": "🐱 Cat",
    "Cow": "🐮 Cow",
    "Deer": "🦌 Deer",
    "Dog": "🐶 Dog",
    "Dolphin": "🐬 Dolphin",
    "Elephant": "🐘 Elephant",
    "Giraffe": "🦒 Giraffe",
    "Horse": "🐴 Horse",
    "Kangaroo": "🦘 Kangaroo",
    "Lion": "🦁 Lion",
    "Panda": "🐼 Panda",
    "Tiger": "🐯 Tiger",
    "Zebra": "🦓 Zebra",
}

labels_ar = {
    "Bear": "🐻 دب",
    "Bird": "🐦 طائر",
    "Cat": "🐱 قطة",
    "Cow": "🐮 بقرة",
    "Deer": "🦌 غزال",
    "Dog": "🐶 كلب",
    "Dolphin": "🐬 دولفين",
    "Elephant": "🐘 فيل",
    "Giraffe": "🦒 زرافة",
    "Horse": "🐴 حصان",
    "Kangaroo": "🦘 كنغر",
    "Lion": "🦁 أسد",
    "Panda": "🐼 باندا",
    "Tiger": "🐯 نمر",
    "Zebra": "🦓 حمار وحشي",
}

# =========================
# CSS (UI)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
* { font-family: 'Tajawal', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #7e22ce);
}
#MainMenu, footer, header { visibility: hidden; }
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}
.result {
    background: rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    lang = st.radio("Language", ["English", "العربية"])
    show_top3 = st.checkbox("Show Top 3", value=True)

    st.markdown("---")
    if os.path.exists(MODEL_PATH):
        model = load_model()
        st.success("Model Loaded")
        st.info(f"TensorFlow {tf.__version__}")
    else:
        st.error("Model file not found")
        st.stop()

labels = labels_en if lang == "English" else labels_ar

# =========================
# SESSION STATE
# =========================
if "image" not in st.session_state:
    st.session_state.image = None

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card" style="text-align:center;">
    <h1>🐾 Animal Image Classifier</h1>
    <p>Deep Learning CNN Model</p>
</div>
""", unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================
c1, c2 = st.columns(2)

# -------- LEFT --------
with c1:
    st.markdown('<div class="card"><h3>📤 Upload Image</h3>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.session_state.image = uploaded.getvalue()

    if st.button("🗑️ Clear"):
        st.session_state.image = None
        st.rerun()

    if st.session_state.image:
        img = Image.open(io.BytesIO(st.session_state.image)).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.info("No image uploaded")

    st.markdown("</div>", unsafe_allow_html=True)

# -------- RIGHT --------
with c2:
    st.markdown('<div class="card"><h3>🧠 Prediction</h3>', unsafe_allow_html=True)

    if st.session_state.image and st.button("🔍 Predict"):
        img = Image.open(io.BytesIO(st.session_state.image)).convert("RGB")
        resized = img.resize(IMG_SIZE)

        x = np.array(resized, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        start = time.time()
        preds = model.predict(x, verbose=0)[0]
        elapsed = (time.time() - start) * 1000

        top_idx = np.argsort(preds)[::-1]
        best = top_idx[0]
        best_name = class_names[int(best)]
        conf = preds[int(best)] * 100

        st.markdown(f"""
        <div class="result">
            <h2>{labels[best_name]}</h2>
            <h3>{conf:.2f}%</h3>
            <p>Inference Time: {elapsed:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf))

        if show_top3:
            st.markdown("### 🏆 Top 3")
            for i in top_idx[:3]:
                st.write(f"{labels[class_names[int(i)]]} — {preds[int(i)]*100:.2f}%")

    else:
        st.info("Upload image and click Predict")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("🚀 Built with TensorFlow & Streamlit")
