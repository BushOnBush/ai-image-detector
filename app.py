import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ==========================
# DEVICE
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# LOAD RESNET50
# ==========================

model = models.resnet50(weights=None)

# Same classifier as training
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# Load trained weights
from huggingface_hub import hf_hub_download


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

model = model.to(device)
model.eval()



# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# ==========================
# PREDICTION FUNCTION
# ==========================

def predict(image):

    if image is None:
        return "Please upload an image."


    image = image.convert("RGB")

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)


    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    confidence = confidence.item() * 100
    predicted = predicted.item()


    # IMPORTANT:
    # Change these labels based on your dataset classes

    classes = [
        "AI Generated Image",
        "Real Image"
    ]


    return (
        f"{classes[predicted]}\n"
        f"Confidence: {confidence:.2f}%"
    )



# ==========================
# GRADIO APP
# ==========================

demo = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="AI Image Detector",

    description=(
        "ResNet50 fine-tuned on AI-generated "
        "and real images."
    )
)


demo.launch()