import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */
    .stApp {
        background: radial-gradient(circle at top, #131b2e 0%, #0b0f1a 65%);
    }

    #MainMenu, footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Hide only the "Deploy" button, not the whole toolbar — the
       toolbar also contains the sidebar expand arrow, and hiding it
       entirely was hiding that arrow too. */
    button[data-testid="stAppDeployButton"],
    div[data-testid="stAppDeployButton"] {
        display: none;
    }

    /* Make the sidebar expand arrow big, obvious, and easy to spot. */
    [data-testid="stSidebarCollapsedControl"] {
        top: 14px;
        left: 14px;
        z-index: 999999;
    }

    [data-testid="stSidebarCollapsedControl"] button {
        transform: scale(1.9);
        transform-origin: top left;
        background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(96, 165, 250, 0.45);
    }

    [data-testid="stSidebarCollapsedControl"] button svg {
        color: #0b0f1a !important;
        fill: #0b0f1a !important;
    }

    /* Also enlarge the collapse arrow once the sidebar is open, so it
       stays consistent. */
    [data-testid="stSidebarCollapseButton"] button {
        transform: scale(1.5);
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 780px;
    }

    /* ---------- Header ---------- */
    .main-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 6px;
        letter-spacing: -1px;
    }

    /* Gradient is applied only to the text span, not the emoji, so the
       emoji keeps its native color glyph instead of rendering as a
       blank tinted box. */
    .main-title .gradient-text {
        background: linear-gradient(90deg, #60a5fa, #a78bfa 60%, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 34px;
        line-height: 1.5;
    }

    /* ---------- Generic surface card ---------- */
    .surface-card {
        padding: 22px 24px;
        border-radius: 16px;
        background: linear-gradient(155deg, #1a2438 0%, #141c2e 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    /* ---------- Result card ---------- */
    .result-card {
        padding: 28px 26px;
        border-radius: 18px;
        margin-top: 22px;
        border: 1px solid var(--accent-border);
        background: linear-gradient(155deg, var(--accent-bg-1) 0%, var(--accent-bg-2) 100%);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .result-label {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        font-size: 25px;
        font-weight: 800;
        color: var(--accent-text);
        margin-bottom: 18px;
        text-align: center;
    }

    /* ---------- Custom progress bar ---------- */
    .progress-outer {
        width: 100%;
        height: 20px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.15);
        overflow: hidden;
        position: relative;
        margin-bottom: 10px;
    }

    .progress-inner {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent-bar-1), var(--accent-bar-2));
        transition: width 0.6s ease;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 4px;
    }

    .confidence-label {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .confidence-value {
        font-size: 22px;
        font-weight: 800;
        color: var(--accent-text);
    }

    .note-pill {
        margin-top: 18px;
        padding: 10px 16px;
        border-radius: 12px;
        font-size: 14px;
        text-align: center;
        background: rgba(148, 163, 184, 0.08);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(155deg, #1a2438 0%, #141c2e 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        padding: 14px 10px;
        border-radius: 14px;
        text-align: center;
    }

    div[data-testid="stMetricLabel"] {
        justify-content: center;
    }

    div[data-testid="stMetricValue"] {
        justify-content: center;
        color: #e2e8f0;
    }

    /* ---------- Uploader ---------- */
    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(155deg, #1a2438 0%, #141c2e 100%);
        border: 1.5px dashed rgba(148, 163, 184, 0.3);
        border-radius: 16px;
    }

    /* ---------- Section headers ---------- */
    h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================
# HEADER
# ==========================

st.markdown(
    '<div class="main-title">🖼️ <span class="gradient-text">AI Image Detector</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Detect AI-generated images using a fine-tuned ResNet50 deep learning model.</div>',
    unsafe_allow_html=True
)


# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application uses **transfer learning with ResNet50**
        to classify images as:

        🤖 AI Generated

        📷 Real Image


        The model was trained on:

        - 30,000 AI-generated images
        - 30,000 real images


        Validation Accuracy: **95.18%**

        Validation Loss: **0.1439**
        """
    )

    st.divider()

    st.header("Model Details")

    st.write(
        """
        **Architecture:** ResNet50

        **Framework:** PyTorch

        **Training:** Transfer Learning

        **Optimizer:** Adam

        **Loss Function:** CrossEntropyLoss

        **Classes:**
        - Fake
        - Real

        **Dataset:** [Kaggle – AI vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
        """
    )


# ==========================
# DEVICE
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model_path = hf_hub_download(
        repo_id="BushOnBush/aiimagedetector",
        filename="best_model.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


with st.spinner("Loading AI detection model..."):
    model = load_model()


# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ["AI Generated Image", "Real Image"]


# ==========================
# MODEL PERFORMANCE
# ==========================

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Validation Accuracy", "95.18%")

with col2:
    st.metric("Validation Loss", "0.1439")

with col3:
    st.metric("Model", "ResNet50")

st.divider()


# ==========================
# IMAGE UPLOAD
# ==========================

st.subheader("🔍 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.convert("RGB")
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    # ==========================
    # PREDICTION
    # ==========================

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    label = classes[prediction.item()]

    # Clamp for safe display/width math, and round once so the bar width
    # and the printed percentage always agree (this is what caused the
    # bar to look "not full" at a displayed 100.00%).
    confidence = min(max(confidence.item(), 0.0), 1.0)
    confidence_pct = round(confidence * 100, 2)
    bar_width = 100.0 if confidence_pct >= 99.95 else confidence_pct

    st.divider()
    st.subheader("Prediction")

    is_ai = prediction.item() == 0

    if is_ai:
        accent_border = "rgba(248, 113, 113, 0.35)"
        accent_bg_1 = "#2a1620"
        accent_bg_2 = "#1c1220"
        accent_text = "#fca5a5"
        accent_bar_1 = "#f87171"
        accent_bar_2 = "#fb7185"
        icon = "🤖"
        label_text = "AI Generated Image"
    else:
        accent_border = "rgba(74, 222, 128, 0.35)"
        accent_bg_1 = "#132a1e"
        accent_bg_2 = "#0f2018"
        accent_text = "#86efac"
        accent_bar_1 = "#4ade80"
        accent_bar_2 = "#22d3ee"
        icon = "📷"
        label_text = "Real Image"

    if confidence_pct >= 90:
        note = "✅ The model is highly confident in this prediction."
    elif confidence_pct >= 70:
        note = "⚠️ The model is moderately confident in this prediction."
    else:
        note = "❔ The model is uncertain — try another image."

    st.markdown(
        f"""
        <div class="result-card" style="--accent-border:{accent_border}; --accent-bg-1:{accent_bg_1}; --accent-bg-2:{accent_bg_2}; --accent-text:{accent_text}; --accent-bar-1:{accent_bar_1}; --accent-bar-2:{accent_bar_2};">
            <div class="result-label">{icon} {label_text}</div>
            <div class="progress-outer">
                <div class="progress-inner" style="width:{bar_width}%;"></div>
            </div>
            <div class="confidence-row">
                <span class="confidence-label">Confidence</span>
                <span class="confidence-value">{confidence_pct:.2f}%</span>
            </div>
            <div class="note-pill">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================
# FOOTER
# ==========================

st.divider()
st.caption("Built with PyTorch • ResNet50 • Streamlit • Hugging Face")