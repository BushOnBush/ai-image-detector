import io
import uuid
from datetime import datetime, timezone

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download, HfApi


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
# FEEDBACK STORAGE (Hugging Face dataset)
# ==========================
# Every "Correct" / "Incorrect" click uploads the image + the true label to
# this dataset repo, alongside metadata.csv. This is a *data collection*
# step only — it does not retrain the model. Periodically download this
# dataset, mix it into your training set, and re-run training to produce a
# new best_model.pth, then upload that to update the live model.
#
# Setup required:
#   1. Create a new dataset repo on huggingface.co, e.g.
#      "BushOnBush/aiimagedetector-feedback" (set repo type to "Dataset").
#   2. Create a HF access token with WRITE permission.
#   3. In Streamlit Cloud → App settings → Secrets, add:
#        HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

FEEDBACK_REPO_ID = "BushOnBush/aiimagedetector-feedback"
FEEDBACK_METADATA_FILE = "metadata.csv"


def get_hf_token():
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def submit_feedback(image, predicted_label, confidence_pct, correct_label):
    """Uploads the image and appends a row to metadata.csv in the feedback
    dataset repo. Returns True on success, False otherwise."""

    token = get_hf_token()

    if not token:
        st.error(
            "Feedback storage isn't configured yet — add an HF_TOKEN "
            "with write access to this app's Streamlit secrets."
        )
        return False

    api = HfApi(token=token)

    feedback_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    image_path_in_repo = f"images/{feedback_id}.png"

    try:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        api.upload_file(
            path_or_fileobj=img_buffer,
            path_in_repo=image_path_in_repo,
            repo_id=FEEDBACK_REPO_ID,
            repo_type="dataset",
            token=token,
        )

        header = "feedback_id,timestamp,predicted_label,confidence_pct,correct_label,image_path\n"

        try:
            existing_path = hf_hub_download(
                repo_id=FEEDBACK_REPO_ID,
                repo_type="dataset",
                filename=FEEDBACK_METADATA_FILE,
                token=token,
            )
            with open(existing_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
        except Exception:
            existing_content = header

        new_row = f"{feedback_id},{timestamp},{predicted_label},{confidence_pct:.2f},{correct_label},{image_path_in_repo}\n"
        updated_content = existing_content + new_row

        api.upload_file(
            path_or_fileobj=io.BytesIO(updated_content.encode("utf-8")),
            path_in_repo=FEEDBACK_METADATA_FILE,
            repo_id=FEEDBACK_REPO_ID,
            repo_type="dataset",
            token=token,
        )

        return True

    except Exception as e:
        st.error(f"Couldn't save feedback: {e}")
        return False


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

    /* ---------- Feedback ---------- */
    .feedback-heading {
        text-align: center;
        font-size: 15px;
        font-weight: 600;
        color: #cbd5e1;
        margin: 24px 0 12px;
    }

    div[data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: linear-gradient(155deg, #1a2438 0%, #141c2e 100%);
        color: #e2e8f0;
        font-weight: 600;
    }

    div[data-testid="stButton"] button:hover {
        border-color: rgba(148, 163, 184, 0.5);
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


        Test Accuracy: **95.18%**

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
    # This checkpoint's head is an nn.Sequential (index 0 = Dropout, index 1 =
    # Linear) rather than a bare nn.Linear — that's why the state dict has
    # "fc.1.weight"/"fc.1.bias" instead of "fc.weight"/"fc.bias". Dropout has
    # no learnable parameters, so its exact probability doesn't affect
    # loading, and eval() mode disables it anyway.
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    model_path = hf_hub_download(
        repo_id="BushOnBush/aiimagedetector2",
        filename="best_model2.pth"
    )

    checkpoint = torch.load(model_path, map_location=device)

    # Checkpoints saved with early stopping / training metadata (e.g. via
    # torch.save({"model_state_dict": ..., "optimizer_state_dict": ...}, ...))
    # come back as a dict wrapping the actual weights, rather than the raw
    # state dict itself. Unwrap it if that's the shape we got; otherwise
    # assume it's already a plain state dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
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
    st.metric("Testing Accuracy", "95.18%")

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
    # FEEDBACK
    # ==========================

    feedback_key = f"feedback_state_{uploaded_file.file_id}"
    submitted_key = f"feedback_submitted_{uploaded_file.file_id}"

    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = None
    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False

    if st.session_state[submitted_key]:
        st.markdown(
            '<div class="note-pill">🙏 Thanks — your feedback was saved and will help retrain the model.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="feedback-heading">Was this prediction correct?</div>', unsafe_allow_html=True)

        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button("✅ Correct", use_container_width=True, key=f"correct_btn_{uploaded_file.file_id}"):
                st.session_state[feedback_key] = "correct"

        with fb_col2:
            if st.button("❌ Incorrect", use_container_width=True, key=f"incorrect_btn_{uploaded_file.file_id}"):
                st.session_state[feedback_key] = "incorrect"

        if st.session_state[feedback_key] == "correct":
            with st.spinner("Saving feedback..."):
                if submit_feedback(image, label_text, confidence_pct, label_text):
                    st.session_state[submitted_key] = True
                    st.rerun()

        elif st.session_state[feedback_key] == "incorrect":
            corrected_label = st.radio(
                "What's the actual label?",
                classes,
                horizontal=True,
                key=f"correction_{uploaded_file.file_id}"
            )

            if st.button("Submit correction", key=f"submit_correction_{uploaded_file.file_id}"):
                with st.spinner("Saving feedback..."):
                    if submit_feedback(image, label_text, confidence_pct, corrected_label):
                        st.session_state[submitted_key] = True
                        st.rerun()


# ==========================
# FOOTER
# ==========================

st.divider()
st.caption("Built with PyTorch • ResNet50 • Streamlit • Hugging Face")