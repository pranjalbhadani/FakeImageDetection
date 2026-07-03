import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import numpy as np
import torch
from PIL import Image

from src.data.transformer import get_val_transform
from src.models.mobilenet import ScreenDetectorMobileNet
from src.utils.helpers import get_project_root

@st.cache_resource
def load_model():
    root = get_project_root()
    weights_path = root / "weights" / "best_model.pth"
    device = torch.device("cpu")
    
    model = ScreenDetectorMobileNet(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    transform = get_val_transform()
    return model, transform, device

st.set_page_config(page_title="Live Screen Detector Demo", page_icon="📷")

st.title("📷 Live Screen Detector")
st.write("Take a picture using your camera, and the model will predict whether it's a real photo or a photo of a screen.")

try:
    model, transform, device = load_model()
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    # Convert image to RGB
    image = Image.open(camera_image).convert("RGB")
    image_np = np.array(image)
    
    # Preprocess
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        
    st.markdown("---")
    st.subheader("Result")
    
    st.write(f"Probability of being a Screen: **{prob:.4f}**")
    
    if prob > 0.5:
        st.error("🚨 This is likely a **PHOTO OF A SCREEN**.")
    else:
        st.success("✅ This is likely a **REAL PHOTO**.")
