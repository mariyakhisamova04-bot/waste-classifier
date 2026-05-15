import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

print("Starting waste sorting app...")

# -----------------------
# MODEL
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
        print("Model file not found")
        return None

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Model loaded")
    return model


model = load_model()

# -----------------------
# CLASSES
# -----------------------
classes = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# -----------------------
# TRANSFORM
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
# UI
# -----------------------
st.title("♻️ Waste Classifier (Streamlit)")
st.write("Загрузите изображение и модель определит класс.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def predict(image: Image.Image):
    if model is None:
        return "Model not loaded"

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        idx = torch.argmax(probs).item()
        confidence = probs[idx].item()

    return classes[idx], confidence * 100


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, conf = predict(image)

    st.subheader("Prediction")
    st.success(f"{label} ({conf:.1f}%)")