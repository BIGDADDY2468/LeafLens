import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="LeafLens | Plant Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --green: #B7F36B;
        --mint: #193522;
        --ink: #F1F8F0;
        --muted: #A8B9AD;
        --panel: #17231B;
        --line: #2D4434;
    }

    * { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 10% 0%, #203B27 0%, transparent 28%), radial-gradient(circle at 90% 20%, #182F22 0%, transparent 30%), linear-gradient(135deg, #0B120E 0%, #101A13 48%, #0A110D 100%);
        color: var(--ink);
    }
    .stApp:before { content:''; position:fixed; inset:0; pointer-events:none; background-image:linear-gradient(rgba(183,243,107,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(183,243,107,.025) 1px, transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom, black, transparent 75%); }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:#B4C5B8; }
    [data-testid="stFileUploader"] section { background:#142119 !important; border:1px dashed #52745A !important; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] label { color:#B4C5B8 !important; }
    [data-testid="stFileUploader"] button { background:#243D2A !important; color:#E8F6E8 !important; border-color:#52745A !important; }
    [data-testid="stAlert"] { background:#18291D !important; border-color:#35563B !important; color:#D7EBDD !important; }
    hr { border-color:#2D4434 !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color:#8EA694 !important; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
    h1 { font-size: clamp(2.4rem, 5vw, 4.8rem) !important; line-height: 1.02 !important; letter-spacing: -0.06em; }
    h2 { letter-spacing: -0.04em; }
    p, li { color: #53645A; font-size: 1.03rem; line-height: 1.7; }

    .block-container { max-width: 1220px; padding-top: 1.2rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { display: none; }

    .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem; }
    .brand { display:flex; align-items:center; gap:.65rem; font-family:'Space Grotesk'; font-size:1.35rem; font-weight:700; color:var(--ink); }
    .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:13px; background:var(--ink); color:var(--green); font-size:1.35rem; box-shadow: 0 8px 24px rgba(16,37,26,.15); }
    .pill { border:1px solid var(--line); color:#597064; background:rgba(255,255,255,.65); padding:.45rem .8rem; border-radius:99px; font-size:.82rem; }

    div[data-testid="stHorizontalBlock"] button { border-radius: 12px !important; border: 1px solid transparent !important; background: transparent !important; color: #A9BDAE !important; font-weight: 600 !important; }
    div[data-testid="stHorizontalBlock"] button:hover { color: #F1F8F0 !important; border-color: #3B5A42 !important; background: #1A2D20 !important; }
    .stButton > button[kind="primary"], div[data-testid="stHorizontalBlock"] button[kind="primary"] { background:linear-gradient(135deg,#B7F36B,#71C85A) !important; color:#102016 !important; border:none !important; font-weight:700 !important; }
    .stButton > button[kind="primary"] p { color:#102016 !important; }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] { background: var(--ink) !important; color: white !important; }

    .hero { padding: 3rem 0 2rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.18em; font-size:.76rem; font-weight:700; color:#4D8B5D; margin-bottom:1rem; }
    .hero-copy { max-width: 650px; }
    .hero-copy p { font-size:1.15rem; max-width:590px; }
    .hero-art { min-height: 300px; border-radius: 34px; background: radial-gradient(circle at 70% 25%, #DAF7A9 0 10%, transparent 11%), radial-gradient(circle at 25% 80%, #C7EDCB 0 18%, transparent 19%), linear-gradient(145deg, #173D26, #2E7142); position:relative; overflow:hidden; box-shadow: 0 20px 60px rgba(24,79,42,.18); }
    .hero-art:before { content:'🌿'; position:absolute; font-size:11rem; right:11%; top:17%; transform:rotate(-16deg); filter:drop-shadow(0 20px 18px rgba(0,0,0,.2)); }
    .hero-art:after { content:'AI-POWERED'; position:absolute; right:10%; bottom:10%; color:#DFF7E5; letter-spacing:.2em; font-size:.7rem; font-weight:700; }

    .card { background:linear-gradient(145deg, rgba(27,45,32,.96), rgba(17,29,21,.96)); border:1px solid rgba(74,111,80,.55); border-radius:24px; padding:1.35rem; box-shadow:0 12px 35px rgba(0,0,0,.25); height:100%; }
    .card h3 { color:#E8F5E9; }
    .card:hover { box-shadow:0 20px 42px rgba(0,0,0,.38); }
    .upload-box { background:rgba(20,37,25,.78); border-color:#52745A; }
    .pill { border-color:#36543D; color:#B2C8B5; background:rgba(26,45,31,.8); }
    .muted { color:#9DB3A1; }
    .card h3 { margin-top:.2rem; font-size:1.2rem; }
    .icon { font-size:1.7rem; }
    .metric { font-family:'Space Grotesk'; font-size:2.2rem; font-weight:700; color:var(--ink); }
    .muted { color:var(--muted); font-size:.9rem; }
    .section-space { height:1.2rem; }
    .upload-box { background:rgba(255,255,255,.7); border:1px dashed #A9C8AE; border-radius:25px; padding:1rem; }
    .result-box { background:linear-gradient(135deg, #173D26, #2F7444); color:white; border-radius:25px; padding:1.6rem; box-shadow:0 15px 40px rgba(24,79,42,.17); }
    .result-box h2, .result-box p, .result-box span { color:white !important; }
    .tag { display:inline-block; padding:.35rem .65rem; border-radius:99px; background:#D8F6B7; color:#28592E; font-size:.78rem; font-weight:700; margin-top:.5rem; }
    .footer { text-align:center; color:#779080; padding:2.5rem 0 1rem; font-size:.85rem; }

    /* Super-cool motion system */
    @keyframes fadeUp { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:translateY(0); } }
    @keyframes floatLeaf { 0%,100% { transform:translateY(0) rotate(-16deg); } 50% { transform:translateY(-18px) rotate(-7deg); } }
    @keyframes drift { 0% { transform:translate(0,0) rotate(0deg); opacity:0; } 15% { opacity:.8; } 80% { opacity:.65; } 100% { transform:translate(90px,-190px) rotate(180deg); opacity:0; } }
    @keyframes glow { 0%,100% { box-shadow:0 0 0 0 rgba(155,225,93,.15); } 50% { box-shadow:0 0 0 14px rgba(155,225,93,0); } }
    @keyframes scan { 0% { top:8%; opacity:0; } 12% { opacity:1; } 88% { opacity:1; } 100% { top:90%; opacity:0; } }
    @keyframes shimmer { 0% { background-position:-500px 0; } 100% { background-position:500px 0; } }
    .hero, .card, .upload-box, .result-box, [data-testid="stImage"] { animation:fadeUp .7s ease both; }
    .card:hover { transform:translateY(-7px) scale(1.01); transition:transform .25s ease, box-shadow .25s ease; box-shadow:0 20px 42px rgba(28,74,42,.13); }
    .hero-art { animation:glow 3s ease-in-out infinite; }
    .hero-art:before { animation:floatLeaf 4s ease-in-out infinite; }
    .hero-art .leaf-particle { position:absolute; color:#D8F6B7; font-size:1.35rem; animation:drift 6s linear infinite; }
    .hero-art .p1 { left:12%; bottom:7%; animation-delay:0s; }
    .hero-art .p2 { left:35%; bottom:2%; animation-delay:2s; font-size:.9rem; }
    .hero-art .p3 { left:55%; bottom:12%; animation-delay:3.7s; font-size:1.7rem; }
    .scan-frame { position:relative; overflow:hidden; border-radius:25px; }
    .scan-frame:after { content:''; position:absolute; left:5%; right:5%; top:8%; height:3px; border-radius:99px; background:linear-gradient(90deg, transparent, #9BE15D, white, #9BE15D, transparent); box-shadow:0 0 18px #9BE15D; animation:scan 2.8s ease-in-out infinite; pointer-events:none; }
    .stButton > button { transition:transform .2s ease, box-shadow .2s ease !important; }
    .stButton > button:hover { transform:translateY(-3px) !important; box-shadow:0 10px 22px rgba(35,91,49,.18) !important; }
    .result-box { animation:fadeUp .6s ease both, glow 3s ease-in-out infinite 1s; }
    .loading-shimmer { height:12px; border-radius:99px; background:linear-gradient(90deg,#e9f5eb 25%,#cdebcf 50%,#e9f5eb 75%); background-size:500px 100%; animation:shimmer 1.5s infinite linear; }
    @media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation:none !important; transition:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model and classes
# -----------------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')


def predict_image(uploaded_file):
    model = load_model()
    image = Image.open(uploaded_file).convert('RGB').resize((128, 128))
    input_arr = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    prediction = model.predict(input_arr, verbose=0)[0]
    index = int(np.argmax(prediction))
    confidence = float(prediction[index]) * 100
    return index, confidence


def pretty_name(label):
    return label.replace('___', ' · ').replace('_', ' ').replace(',', ', ')

# -----------------------------
# Navigation
# -----------------------------
st.markdown('<div class="topbar"><div class="brand"><div class="brand-mark">✦</div>LeafLens</div><div class="pill">Better care, better growth.</div></div>', unsafe_allow_html=True)

nav_left, nav_center, nav_right = st.columns([1, 1, 1])
with nav_left:
    home_btn = st.button('⌂  Home', use_container_width=True)
with nav_center:
    scan_btn = st.button('◉  Disease Scanner', type='primary', use_container_width=True)
with nav_right:
    about_btn = st.button('◎  About', use_container_width=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if home_btn: st.session_state.page = 'Home'
if scan_btn: st.session_state.page = 'Disease Scanner'
if about_btn: st.session_state.page = 'About'

page = st.session_state.page
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

# -----------------------------
# Home
# -----------------------------
if page == 'Home':
    left, right = st.columns([1.12, .88], gap='large')
    with left:
        st.markdown('<div class="hero"><div class="eyebrow">Smart crop care • powered by AI</div>', unsafe_allow_html=True)
        st.markdown('<h1>See what your<br><span style="color:#4D8B5D">leaves are saying.</span></h1>', unsafe_allow_html=True)
        st.markdown('<p>Upload a clear leaf photo and get an instant, intelligent health check for your crops.</p></div>', unsafe_allow_html=True)
        if st.button('Start a free scan  →', type='primary'):
            st.session_state.page = 'Disease Scanner'
            st.rerun()
    with right:
        st.markdown('<div class="hero-art"><span class="leaf-particle p1">✦</span><span class="leaf-particle p2">·</span><span class="leaf-particle p3">✧</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap='medium')
    cards = [
        ('01', 'Upload', 'Add a well-lit image of the affected plant leaf.'),
        ('02', 'Analyze', 'Our trained vision model checks visual symptoms.'),
        ('03', 'Act', 'Use the result as a starting point for next steps.'),
    ]
    for col, (num, title, text) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(f'<div class="card"><div class="muted">{num}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

# -----------------------------
# Disease scanner
# -----------------------------
elif page == 'Disease Scanner':
    st.markdown('<div class="eyebrow">Live diagnosis workspace</div>', unsafe_allow_html=True)
    st.title('Disease scanner')
    st.write('Give LeafLens a close, clear view of a single leaf. For best results, avoid shadows and blurry images.')

    upload_col, preview_col = st.columns([.9, 1.1], gap='large')
    with upload_col:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('Choose a leaf image', type=['jpg', 'jpeg', 'png'], label_visibility='visible')
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded_file:
            if st.button('Analyze image  ✦', type='primary', use_container_width=True):
                with st.spinner('Reading leaf patterns...'):
                    try:
                        result_index, confidence = predict_image(uploaded_file)
                        st.session_state.result = (result_index, confidence)
                    except Exception as error:
                        st.error(f'Could not analyze this image: {error}')
        else:
            st.info('Upload an image to unlock the scanner.')

    with preview_col:
        if uploaded_file:
            st.image(uploaded_file, caption='Uploaded leaf', use_container_width=True)
        else:
            st.markdown('<div class="card" style="min-height:260px; display:grid; place-items:center; text-align:center;">'
            '<div><div style="font-size:3rem">🍃</div><h3>Your preview appears here</h3>'
            '<p class="muted">Supported formats: JPG, JPEG and PNG</p></div></div>', unsafe_allow_html=True)

    if 'result' in st.session_state:
        result_index, confidence = st.session_state.result
        label = pretty_name(CLASS_NAMES[result_index])
        status = 'Healthy leaf' if 'healthy' in CLASS_NAMES[result_index].lower() else 'Possible disease detected'
        st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box"><div class="muted" style="color:#BDE5C5 !important">ANALYSIS RESULT</div><h2>{label}</h2><span class="tag">{status}</span><p style="margin-top:1rem">Model confidence: <strong>{confidence:.1f}%</strong></p></div>', unsafe_allow_html=True)
        st.caption('This tool is for educational screening and should not replace advice from an agricultural professional.')

# -----------------------------
# About
# -----------------------------
else:
    st.markdown('<div class="eyebrow">Behind the lens</div>', unsafe_allow_html=True)
    st.title('Built for healthier harvests.')
    st.write('LeafLens uses a TensorFlow image-classification model to recognize visual patterns across common crop leaf conditions.')
    a, b = st.columns([1.3, .7], gap='large')
    with a:
        st.markdown('<div class="card"><h3>About the dataset</h3><p>The model is trained on a PlantVillage-style dataset containing approximately ' \
        '87,000 RGB images of healthy and diseased crop leaves across 38 classes. The data is organized into training, validation and test sets.</p>'
        '<h3>How to get better results</h3><ul><li>Photograph one leaf at a time.</li><li>Use natural, even lighting.</li><li>Keep the leaf in focus and centered.</li>'
        '<li>Use the prediction as an early screening signal.</li></ul></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="metric">38</div><div class="muted">recognition classes</div><hr><div class="metric">128²'
        '</div><div class="muted">image input size</div><hr><div class="metric">AI</div><div class="muted">visual pattern analysis</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">LeafLens · A smarter first look at plant health 🌱</div>', unsafe_allow_html=True)
