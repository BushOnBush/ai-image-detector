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
    page_icon="🖼️"
)

st.title("🖼️ AI Image Detector")
st.write(
    "Upload an image and the ResNet50 model will predict "
    "whether it is AI-generated or real."
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

    # Create ResNet50
    model = models.resnet50(
        weights=None
    )


    # Match training classifier
    # model.fc = nn.Linear(model.fc.in_features, 2)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )


    # Download trained weights
    model_path = hf_hub_download(
        repo_id="BushOnBush/aiimagedetector",
        filename="best_model.pth"
    )


    # Load weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )


    model.to(device)
    model.eval()


    return model



with st.spinner("Loading AI model..."):
    model = load_model()



# ==========================
# IMAGE TRANSFORMATION
# ==========================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])



# ==========================
# CLASS LABELS
# ==========================

# From training:
# {'fake': 0, 'real': 1}

classes = [
    "AI Generated Image",
    "Real Image"
]



# ==========================
# IMAGE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # Convert image to RGB
    image = image.convert(
        "RGB"
    )


    # Preprocess
    img = transform(
        image
    )


    # Add batch dimension
    img = img.unsqueeze(
        0
    )


    img = img.to(device)



    # ==========================
    # PREDICTION
    # ==========================

    with torch.no_grad():

        output = model(
            img
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    predicted_class = classes[
        prediction.item()
    ]

    confidence = confidence.item() * 100



    # ==========================
    # DISPLAY RESULT
    # ==========================

    st.subheader("Prediction")


    if prediction.item() == 0:

        st.error(
            f"🤖 {predicted_class}"
        )

    else:

        st.success(
            f"✅ {predicted_class}"
        )


    st.write(
        f"Confidence: {confidence:.2f}%"
    )