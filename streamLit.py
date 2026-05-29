import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import time

from model import UNetMulti, DoubleConv

MODEL_PATH = 'unet_best.pth'

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetMulti()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess(image: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    pil_image = Image.fromarray(image)
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def run_inference(model, device, image_tensor: torch.Tensor):
    image_tensor = image_tensor.to(device)
    t_start = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output)
        masks = (probs > 0.5).squeeze(0).cpu().numpy()
    elapsed_ms = (time.time() - t_start) * 1000
    mask_liver = masks[0].astype(np.uint8) * 255
    mask_hem = masks[1].astype(np.uint8) * 255
    return mask_liver, mask_hem, elapsed_ms

def overlay(image: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    result = image.copy()
    result[mask > 0] = color
    return cv2.addWeighted(image, 0.7, result, 0.3, 0)

st.set_page_config(
    page_title="Сегментация гемангиомы печени",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #1F1F1F;
}
h1, h2, h3 {
    color: #0B5ED7;
    font-family: Arial;
}
.stButton > button {
    background-color: #0B5ED7;
    color: white;
    border-radius: 6px;
    border: none;
}
.stButton > button:hover {
    background-color: #084298;
    color: white;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.metric-box {
    background-color: #f0f4ff;
    border-left: 4px solid #0B5ED7;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Сегментация гемангиомы печени")

try:
    model, device = load_model()
    device_name = 'GPU (CUDA)' if device.type == 'cuda' else 'CPU'
    st.caption(f"Модель загружена · Устройство: {device_name}")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.subheader("Загрузка изображения")

uploaded_file = st.file_uploader(
    "Загрузка изображения",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    image_tensor = preprocess(image_np)

    with st.spinner("Выполняется сегментация..."):
        mask_liver, mask_hem, elapsed_ms = run_inference(model, device, image_tensor)

    mask_liver = cv2.resize(mask_liver, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_hem = cv2.resize(mask_hem, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    liver_vis = overlay(image_np, mask_liver, (0, 255, 0))
    hem_vis = overlay(image_np, mask_hem, (255, 0, 0))
    combined_vis = overlay(liver_vis, mask_hem, (255, 0, 0))

    st.markdown(
        f'<div class="metric-box">⏱ Время инференса: <b>{elapsed_ms:.1f} мс</b> · Устройство: <b>{device_name}</b></div>',
        unsafe_allow_html=True
    )

    st.subheader("Результаты")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("### Оригинал")
            st.image(image_np, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Печень")
            st.image(liver_vis, use_container_width=True)

    with col3:
        with st.container(border=True):
            st.markdown("### Гемангиома")
            st.image(hem_vis, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.markdown("### Итог")
            st.image(combined_vis, use_container_width=True)