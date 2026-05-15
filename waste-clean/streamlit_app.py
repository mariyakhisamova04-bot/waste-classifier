import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from datetime import datetime

# -----------------------
# СТРАНИЦА НАСТРОЕК
# -----------------------
st.set_page_config(
    page_title="Классификатор отходов",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="auto"
)

# -----------------------
# СТИЛИ CSS (русский язык, цвета)
# -----------------------
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .result-success {
        background-color: #D4EDDA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 6px solid #28A745;
        font-size: 1.2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #95A5A6;
        border-top: 1px solid #ECF0F1;
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# ЗАГРУЗКА МОДЕЛИ (кэш)
# -----------------------
@st.cache_resource
def load_model():
    print("Loading model...")

    num_classes = 10
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model_path = "best_classifier.pth"

    if not os.path.exists(model_path):
        st.error("❌ Файл модели 'best_classifier.pth' не найден!")
        return None

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Model loaded")
    return model

model = load_model()

# -----------------------
# КЛАССЫ (на русском + англ. для отладки)
# -----------------------
classes_ru = [
    'батарейка',
    'биологические отходы',
    'картон',
    'одежда',
    'стекло',
    'металл',
    'бумага',
    'пластик',
    'обувь',
    'смешанный мусор'
]

classes_en = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# ----- РЕКОМЕНДАЦИИ ПО УТИЛИЗАЦИИ -----
recommendations = {
    'батарейка': '🔋 Опасные отходы. Сдать в специальный пункт приёма батареек.',
    'биологические отходы': '🍌 Пищевые отходы. На компост или в отдельный контейнер для органики.',
    'картон': '📦 Картон. В контейнер для бумаги и картона (сплющить).',
    'одежда': '👕 Текстиль. В контейнер для одежды или секонд-хенд.',
    'стекло': '🥃 Стекло. В контейнер для стекла (желательно мытое, без крышек).',
    'металл': '🥫 Металл. В контейнер для металла и жести.',
    'бумага': '📄 Бумага. В контейнер для макулатуры.',
    'пластик': '🧴 Пластик. В контейнер для пластика (чистый, сжатый).',
    'обувь': '👟 Обувь. В контейнер для текстиля и обуви.',
    'смешанный мусор': '🗑️ Смешанные отходы. В общий контейнер для неперерабатываемого мусора.'
}

# -----------------------
# ТРАНСФОРМАЦИЯ ИЗОБРАЖЕНИЯ
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# -----------------------
def predict(image: Image.Image):
    if model is None:
        return None, None

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        idx = torch.argmax(probs).item()
        confidence = probs[idx].item()

    return classes_ru[idx], confidence

# -----------------------
# БОКОВАЯ ПАНЕЛЬ (фильтры, инфо)
# -----------------------
with st.sidebar:
    st.header("ℹ️ О системе")
    st.markdown("""
    **Разработка:** Система автоматической классификации бытовых отходов  
    **Модель:** ResNet50 (предобучена на ImageNet)  
    **Классы:** 10 типов отходов  
    **Точность модели:** 86.2%  
    """)

    st.markdown("---")
    st.header("📚 Поддерживаемые классы")
    df_classes = pd.DataFrame({
        "Тип отхода": classes_ru,
        "Рекомендация": [recommendations[c][:60] + "..." for c in classes_ru]
    })
    st.dataframe(df_classes, use_container_width=True)

    st.markdown("---")
    st.caption(f"© 2025 Дипломный проект\nОбновлено: {datetime.now().strftime('%d.%m.%Y')}")

# -----------------------
# ОСНОВНОЙ ИНТЕРФЕЙС
# -----------------------
st.markdown('<div class="main-header">♻️ Классификатор отходов</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Загрузите фотографию мусора — нейросеть определит тип отхода и даст рекомендацию</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Выберите изображение (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Поддерживаются форматы: JPG, JPEG, PNG"
)

# Кнопка очистки
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    clear_btn = st.button("🗑️ Очистить", use_container_width=True)

if clear_btn:
    uploaded_file = None
    st.rerun()

# -----------------------
# ОБРАБОТКА ЗАГРУЖЕННОГО ИЗОБРАЖЕНИЯ
# -----------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Две колонки: слева фото, справа результат
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Загруженное изображение", use_container_width=True)

    with col_res:
        with st.spinner("🔍 Анализируем изображение..."):
            label, confidence = predict(image)

        if label is not None:
            # Цветной блок результата
            st.markdown(f"""
            <div class="result-success">
                <strong>📊 Результат:</strong> {label.upper()}<br>
                <strong>🎯 Уверенность:</strong> {confidence*100:.1f}%<br>
                <strong>💡 Рекомендация:</strong> {recommendations.get(label, 'Сортировать согласно местным правилам')}
            </div>
            """, unsafe_allow_html=True)

            # Прогресс-бар уверенности
            st.progress(confidence, text=f"Уверенность модели: {confidence*100:.1f}%")

        else:
            st.error("❌ Ошибка: модель не загружена")

    # Дополнительная информация о классе
    with st.expander("🔍 Подробнее о классе и переработке"):
        st.markdown(f"""
        **Класс:** {label}  
        **Рекомендация по утилизации:** {recommendations.get(label, '—')}  

        **Совет:**  
        - Перед выбросом по возможности очистите отходы от остатков пищи.  
        - Сплющивайте пластиковые бутылки и картонные коробки для экономии места.  
        - Батарейки и лампочки нельзя выбрасывать в обычный мусор — найдите ближайший пункт приёма.
        """)

else:
    st.info("📸 Загрузите изображение, чтобы начать классификацию")

# -----------------------
# ФУТЕР
# -----------------------
st.markdown('<div class="footer">Дипломная работа: Разработка системы автоматической классификации бытовых отходов на основе методов компьютерного зрения</div>', unsafe_allow_html=True)