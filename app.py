import streamlit as st
import torch
import os
from PIL import Image
import yaml
from src.model import DynamicCNN
from src.data_setup import get_train_test_transforms, get_mean_and_std
from src. utils import get_class_names
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "yassinkina/pokemon-cnn"
SAMPLES_DIR = "samples/"
FILENAME = "pokemon_cnn_best.pth"
CONFIG_PATH = "config/config.yaml"

# --- CALLBACKS FOR MUTUAL EXCLUSION ---
def on_upload_change():
    # If a file is uploaded, clear the sample selector
    st.session_state.selector_key += 1

def on_sample_change():
    # If a sample is selected, clear the file uploader
    st.session_state.uploader_key += 1

# --- SESSION STATE INITIALIZATION ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "selector_key" not in st.session_state:
    st.session_state.selector_key = 0

# --- MODEL LOADING ---

@st.cache_resource
def load_model(repo_id: str, filename: str, config_path: str):
    """
    Downloads weights from Hugging Face Hub and initializes the DynamicCNN.
    """
    device = torch.device("cpu") # Streamlit Cloud uses CPU
    
    # 1. Download the weights file to a local cache
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # 2. Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3. Initialize architecture
    model = DynamicCNN(
        n_layers=cfg['model']['n_layers'],
        n_filters=cfg['model']['n_filters'],
        kernel_sizes=cfg['model']['kernel_sizes'],
        dropout_rate=cfg['model']['dropout_rate'],
        fc_size=cfg['model']['fc_size'],
        num_classes=cfg["training"]["num_classes"]
    ).to(device)

    # 4. Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both full checkpoints and state_dict-only files
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.eval()
    return model, device

model, DEVICE = load_model(REPO_ID, FILENAME, CONFIG_PATH)
CLASS_NAMES = get_class_names()
mean, std = get_mean_and_std()
_, test_transform = get_train_test_transforms(mean=mean, std=std)

# --- UI LAYOUT ---
st.set_page_config(page_title="Pokémon Classifier", layout="wide")
st.title("⚡ Pokémon Classifier")
st.markdown("---")

# --- SIDEBAR & MAIN INPUT ---
col_sidebar, col_main = st.columns([1, 3])

with st.sidebar:
    st.header("🧪 Quick Test")
    sample_images = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', ".avif"))] if os.path.exists(SAMPLES_DIR) else []
    
    # Selecting a sample triggers on_sample_change callback
    selected_sample = st.selectbox(
        "Choose a sample:", 
        ["None"] + sample_images,
        key=f"sample_select_{st.session_state.selector_key}",
        on_change=on_sample_change
    )

with col_main:
    # Uploading a file triggers on_upload_change callback
    uploaded_file = st.file_uploader(
        "Or upload your own image...", 
        type=["jpg", "png", "jpeg", "webp", "avif"], 
        key=f"file_up_{st.session_state.uploader_key}",
        on_change=on_upload_change
    )

# --- FINAL IMAGE SELECTION ---
active_img = None
if uploaded_file:
    active_img = Image.open(uploaded_file).convert("RGB")
elif selected_sample != "None":
    active_img = Image.open(os.path.join(SAMPLES_DIR, selected_sample)).convert("RGB")

# --- UI DISPLAY & PREDICTION ---
if active_img:
    if st.button("🗑️ Remove Photo", use_container_width=True):
        st.session_state.uploader_key += 1
        st.session_state.selector_key += 1
        st.rerun()

    st.markdown("---")
    col_img, col_pred = st.columns(2)
    
    with col_img:
        st.image(active_img, caption="Active Image", use_container_width=True)
    
    with col_pred:
        img_tensor = test_transform(active_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_p, top5_i = torch.topk(probs, 5)

        st.success(f"### Prediction: **{CLASS_NAMES[top5_i[0]]}**")
        st.metric("Confidence", f"{top5_p[0].item()*100:.1f}%")
        
        st.write("#### Top 5 Candidates")
        for i in range(5):
            st.write(f"**{CLASS_NAMES[top5_i[i]]}** ({top5_p[i].item()*100:.1f}%)")
            st.progress(top5_p[i].item())
else:
    st.info("👈 Select a sample from the sidebar or upload an image to begin.")