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
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f7fb;
        margin-bottom: 15px;
    }

    .confidence {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
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
    '<div class="subtitle">A ResNet50 deep learning model for detecting AI-generated images.</div>',
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

        The model was trained on **60,000 images**
        consisting of:

        - 30,000 AI-generated images
        - 30,000 real images

        Validation Accuracy:
        **95.18%**

        Validation Loss:
        **0.1439**
        """
    )


    st.divider()


    st.header("Model Information")

    st.write(
        """
        **Architecture:** ResNet50

        **Framework:** PyTorch

        **Training Method:** Transfer Learning

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
# PERFORMANCE SECTION
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
        "Architecture",
        "ResNet50"
    )



st.divider()



# ==========================
# UPLOAD
# ==========================

st.subheader("🔍 Analyze an Image")


uploaded_file = st.file_uploader(
    "Upload an image",
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
        <div class="card">

        <div class="confidence">
        Confidence: {confidence*100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )



    # Confidence explanation

    if confidence > 0.9:

        st.info(
            "The model is highly confident in this prediction."
        )

    elif confidence > 0.7:

        st.warning(
            "The model is moderately confident. Results may vary on unfamiliar images."
        )

    else:

        st.warning(
            "The model is uncertain. Consider testing with another image."
        )



# ==========================
# FUTURE FEATURES
# ==========================

st.divider()

st.subheader("🚀 Future Improvements")

st.write(
    """
    Planned improvements:

    - Grad-CAM visualization to show which regions influenced predictions
    - Larger and more diverse datasets
    - Comparison with Vision Transformer models
    - Detection of specific AI image generators
    """
)



# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Built with PyTorch • ResNet50 • Streamlit • Hugging Face"
)