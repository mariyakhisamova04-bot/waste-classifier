import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gdown

# -----------------------
# 1. ПОДГОТОВКА
# -----------------------
st.set_page_config(
    page_title="Классификатор отходов",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FFFFFF, #80D4F0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
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
        border-color: rgba(255, 255, 255, 0.25);
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
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00C8E8, #0096D8);
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.3);
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
    hr {
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------
# 2. ЗАГРУЗЧИК МОДЕЛИ (С GOOGLE DRIVE)
# -----------------------
@st.cache_resource
def load_model():
    # ID файла из вашей ссылки (скопирован из URL)
    file_id = "1WXKqzz213LvWBXmSXw66lg6Wc46pfrqZ"
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "best_classifier.pth"

    # Скачиваем, если файла нет
    if not os.path.exists(model_path):
        with st.spinner("Загрузка модели. Пожалуйста, подождите..."):
            gdown.download(url, model_path, quiet=False)
        st.success("Модель загружена успешно!")

    # Инициализация модели
    num_classes = 10
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# -----------------------
# 3. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И КЛАССОВ
# -----------------------
model = load_model()

classes_ru = [
    'батарейка', 'биологические отходы', 'картон', 'одежда', 'стекло',
    'металл', 'бумага', 'пластик', 'обувь', 'смешанный мусор'
]

recommendations = {
    'батарейка': 'Опасные отходы. Сдать в специальный пункт приёма батареек.',
    'биологические отходы': 'Пищевые отходы. На компост или в отдельный контейнер.',
    'картон': 'Картон. В контейнер для бумаги и картона (сплющить).',
    'одежда': 'Текстиль. В контейнер для одежды или секонд-хенд.',
    'стекло': 'Стекло. В контейнер для стекла (мытое, без крышек).',
    'металл': 'Металл. В контейнер для металла и жести.',
    'бумага': 'Бумага. В контейнер для макулатуры.',
    'пластик': 'Пластик. В контейнер для пластика (чистый, сжатый).',
    'обувь': 'Обувь. В контейнер для текстиля и обуви.',
    'смешанный мусор': 'Смешанные отходы. В общий контейнер.'
}

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
# 4. ИНТЕРФЕЙС (ШАПКА, КАРТОЧКИ, ТАБЫ)
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
        <span>Главная</span>
        <span>О проекте</span>
        <span>Классы</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Система классификации отходов на основе компьютерного зрения</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Загрузите фотографию или сделайте снимок с веб-камеры. Нейросеть определит тип отхода и даст рекомендацию по утилизации.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Загрузка файла</div>
        <div class="feature-desc">Поддержка JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Веб-камера</div>
        <div class="feature-desc">Снимок в реальном времени</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">10 классов отходов</div>
        <div class="feature-desc">Точность модели 86.2%</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Загрузить файл", "Сделать снимок с веб-камеры"])

with tab1:
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Загруженное изображение", use_container_width=True)
        with col_res:
            with st.spinner("Анализ..."):
                label, conf = predict(image)
            if label:
                st.markdown(f"""
                <div class="result-glass">
                    <div class="result-label">{label.upper()}</div>
                    <div class="result-confidence">Уверенность: {conf*100:.1f}%</div>
                    <div class="result-recommendation">Рекомендация: {recommendations.get(label, 'Сортировать по местным правилам')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf, text=f"Уверенность модели: {conf*100:.1f}%")

with tab2:
    st.markdown("Нажмите на кнопку ниже, чтобы активировать камеру и сделать снимок.")
    camera_image = st.camera_input("Сделать снимок", key="camera_input")
    if camera_image:
        image_cam = Image.open(camera_image)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image_cam, caption="Снимок с камеры", use_container_width=True)
        with col_res:
            with st.spinner("Анализ..."):
                label_cam, conf_cam = predict(image_cam)
            if label_cam:
                st.markdown(f"""
                <div class="result-glass">
                    <div class="result-label">{label_cam.upper()}</div>
                    <div class="result-confidence">Уверенность: {conf_cam*100:.1f}%</div>
                    <div class="result-recommendation">Рекомендация: {recommendations.get(label_cam, 'Сортировать по местным правилам')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf_cam, text=f"Уверенность модели: {conf_cam*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="custom-footer">
    Дипломная работа: Разработка системы автоматической классификации бытовых отходов на основе методов компьютерного зрения<br>
    Модель: ResNet50 | Классов: 10 | Точность: 86.2%
</div>
""", unsafe_allow_html=True)
