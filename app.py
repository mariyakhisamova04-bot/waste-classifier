import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gdown

# -----------------------
# Загрузка модели из Google Drive
# -----------------------
MODEL_URL = "https://drive.google.com/uc?id=1WXKqzz213LvWBXmSXw66lg6Wc46pfrqZ"
MODEL_PATH = "best_classifier.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading model. Please wait..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model loaded successfully")

download_model()

# -----------------------
# Страница настроек
# -----------------------
st.set_page_config(
    page_title="Waste Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# CSS styles
# -----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0A192F 0%, #0D2A3A 100%); }
    .glass-panel {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FFFFFF, #80D4F0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 2rem;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.08);
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #80D4F0;
        margin-top: 0.5rem;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.5rem;
    }
    .result-glass {
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #80D4F0;
    }
    .result-confidence {
        font-size: 1rem;
        color: #FFD966;
    }
    .result-recommendation {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00B4D8, #0077B6);
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        color: white;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00C8E8, #0096D8);
        transform: scale(1.02);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        border-radius: 0;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #80D4F0;
        border-bottom: 2px solid #80D4F0;
    }
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    hr { margin: 1rem 0; border-color: rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Загрузка модели
# -----------------------
@st.cache_resource
def load_model():
    print("Loading ResNet50 model...")
    num_classes = 10
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found")
        return None

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("Model loaded")
    return model

model = load_model()

# -----------------------
# Classes and recommendations
# -----------------------
classes_ru = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

recommendations = {
    'battery': 'Hazardous waste. Dispose at a battery collection point.',
    'biological': 'Organic waste. Compost or separate bio-bin.',
    'cardboard': 'Cardboard. Recycling bin for paper and cardboard.',
    'clothes': 'Textiles. Clothing collection container or second-hand.',
    'glass': 'Glass. Glass recycling bin (rinsed, without lids).',
    'metal': 'Metal. Metal/tin recycling container.',
    'paper': 'Paper. Paper recycling bin.',
    'plastic': 'Plastic. Plastic recycling bin (clean and compressed).',
    'shoes': 'Shoes. Textile/shoe collection container.',
    'trash': 'Mixed waste. General waste container.'
}

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if model is None:
        return None, None
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        idx = torch.argmax(probs).item()
        conf = probs[idx].item()
    return classes_ru[idx], conf

# -----------------------
# Header
# -----------------------
col_logo, col_nav = st.columns([1, 3])
with col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 2rem;">EcoSort AI</span>
    </div>
    """, unsafe_allow_html=True)
with col_nav:
    st.markdown("""
    <div style="display: flex; justify-content: flex-end; gap: 2rem; color: rgba(255,255,255,0.7);">
        <span>Home</span>
        <span>About</span>
        <span>Classes</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Hero section
# -----------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Waste Classification System based on Computer Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a photo or take a picture with your webcam. The neural network will identify the waste type and provide a recycling recommendation.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="feature-card"><div class="feature-title">File Upload</div><div class="feature-desc">JPG, JPEG, PNG</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><div class="feature-title">Webcam</div><div class="feature-desc">Real-time capture</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card"><div class="feature-title">10 waste classes</div><div class="feature-desc">Model accuracy 86.2%</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Main: tabs (file / webcam)
# -----------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Upload File", "Take a picture with webcam"])

with tab1:
    uploaded_file = st.file_uploader("Select image", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col_res:
            with st.spinner("Analyzing..."):
                label, conf = predict(image)
            if label:
                st.markdown(f"""
                <div class="result-glass">
                    <div class="result-label">{label.upper()}</div>
                    <div class="result-confidence">Confidence: {conf*100:.1f}%</div>
                    <div class="result-recommendation">Recommendation: {recommendations.get(label, 'Sort according to local rules')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf, text=f"Model confidence: {conf*100:.1f}%")

with tab2:
    st.markdown("Press the button below to activate the camera and take a picture.")
    camera_image = st.camera_input("Take a picture", key="camera_input")
    if camera_image:
        image_cam = Image.open(camera_image)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image_cam, caption="Camera shot", use_container_width=True)
        with col_res:
            with st.spinner("Analyzing..."):
                label_cam, conf_cam = predict(image_cam)
            if label_cam:
                st.markdown(f"""
                <div class="result-glass">
                    <div class="result-label">{label_cam.upper()}</div>
                    <div class="result-confidence">Confidence: {conf_cam*100:.1f}%</div>
                    <div class="result-recommendation">Recommendation: {recommendations.get(label_cam, 'Sort according to local rules')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf_cam, text=f"Model confidence: {conf_cam*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("""
<div class="custom-footer">
    Diploma thesis: Development of an automatic household waste classification system based on computer vision methods<br>
    Model: ResNet50 | Classes: 10 | Accuracy: 86.2%
</div>
""", unsafe_allow_html=True)
