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
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #f5f7fb;
        margin-top: 20px;
    }

    .confidence {
        font-size: 22px;
        font-weight: bold;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
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
    '<div class="subtitle">Detect whether an image was created by AI using a fine-tuned ResNet50 deep learning model.</div>',
    unsafe_allow_html=True
)



# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application uses transfer learning with
        **ResNet50** to classify images as:

        🤖 AI Generated  
        📷 Real Image
        """
    )


    st.divider()

    st.write("### Model Details")

    st.write(
        """
        **Architecture:** ResNet50  
        **Framework:** PyTorch  
        **Training:** Transfer Learning  
        **Classes:** Fake / Real  
        """
    )



# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# TRANSFORM
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
# UPLOAD
# ==========================

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "",
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


    image = image.convert("RGB")


    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)



    with torch.no_grad():

        output = model(tensor)

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
        float(confidence)
    )


    st.markdown(
        f"""
        <div class="result-card">

        <div class="confidence">
        Confidence: {confidence*100:.2f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )



# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Built with PyTorch • ResNet50 • Streamlit • Hugging Face"
)