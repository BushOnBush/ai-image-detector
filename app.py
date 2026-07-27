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
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    # Two classes:
    # 0 = Real
    # 1 = AI Generated
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
            map_location=torch.device("cpu")
        )
    )


    model.eval()

    return model



model = load_model()



# ==========================
# IMAGE PREPROCESSING
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

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
# PREDICTION FUNCTION
# ==========================

def predict(image):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)


    with torch.no_grad():

        output = model(image_tensor)


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    confidence = confidence.item() * 100


    if predicted.item() == 1:
        result = "AI Generated"

    else:
        result = "Real Image"


    return result, confidence



# ==========================
# APP UI
# ==========================


st.title("🖼️ AI Image Detector")

st.write(
    "Upload an image to determine whether it is AI-generated or real."
)


st.divider()



uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)



if uploaded_file is not None:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    st.divider()


    with st.spinner("Analyzing image..."):

        prediction, confidence = predict(image)



    # ==========================
    # PREDICTION BOX
    # ==========================

    st.subheader("Prediction")


    if prediction == "AI Generated":

        st.error(
            "🤖 AI Generated"
        )

    else:

        st.success(
            "📷 Real Image"
        )



    # ==========================
    # CONFIDENCE BOX
    # ==========================

    st.subheader("Confidence")


    st.info(
        f"{confidence:.2f}%"
    )


    st.caption(
        "Higher confidence means the model is more certain about its prediction."
    )