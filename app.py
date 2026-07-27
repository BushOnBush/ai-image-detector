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
    layout="centered"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }


    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 18px;
        margin-bottom: 30px;
    }


    .confidence-card {

        padding: 20px;
        border-radius: 15px;
        background-color: #1e293b;
        margin-top: 20px;

    }


    .confidence-text {

        font-size: 26px;
        font-weight: bold;
        text-align: center;
        color: white;

    }


    .metric-card {

        padding: 15px;
        border-radius: 12px;
        background-color: #1e293b;

    }


    </style>
    """,
    unsafe_allow_html=True
)



# ==========================
# HEADER
# ==========================

st.markdown(
    '<div class="main-title">🖼️ AI Image Detector</div>',
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


        Validation Accuracy:
        **95.18%**


        Validation Loss:
        **0.1439**
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
        """
    )



# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():

    model = models.resnet50(
        weights=None
    )


    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )


    model_path = hf_hub_download(
        repo_id="BushOnBush/aiimagedetector",
        filename="best_model.pth"
    )


    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )


    model.to(device)
    model.eval()


    return model



with st.spinner("Loading AI detection model..."):

    model = load_model()



# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )

])



classes = [
    "AI Generated Image",
    "Real Image"
]



# ==========================
# MODEL PERFORMANCE
# ==========================

st.subheader("📊 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Validation Accuracy",
        "95.18%"
    )


with col2:

    st.metric(
        "Validation Loss",
        "0.1439"
    )


with col3:

    st.metric(
        "Model",
        "ResNet50"
    )



st.divider()



# ==========================
# IMAGE UPLOAD
# ==========================

st.subheader("🔍 Upload Image")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)



if uploaded_file:


    image = Image.open(
        uploaded_file
    )


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    image = image.convert(
        "RGB"
    )


    tensor = transform(
        image
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(device)



    # ==========================
    # PREDICTION
    # ==========================

    with torch.no_grad():

        output = model(
            tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )



    label = classes[
        prediction.item()
    ]


    confidence = confidence.item()



    st.divider()

    st.subheader("Prediction")



    if prediction.item() == 0:

        st.error(
            "🤖 AI Generated Image"
        )

    else:

        st.success(
            "📷 Real Image"
        )



    st.progress(
        confidence
    )


    st.markdown(
        f"""
        <div class="confidence-card">

        <div class="confidence-text">

        Confidence: {confidence*100:.2f}%

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )



    if confidence > 0.9:

        st.info(
            "The model is highly confident in this prediction."
        )

    elif confidence > 0.7:

        st.warning(
            "The model is moderately confident in this prediction."
        )

    else:

        st.warning(
            "The model is uncertain. Try another image."
        )



# ==========================
# FOOTER
# ==========================

st.divider()


st.caption(
    "Built with PyTorch • ResNet50 • Streamlit • Hugging Face"
)