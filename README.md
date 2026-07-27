---
title: AI Image Detector
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# AI Image Detector

A ResNet50-based deep learning model that classifies images as:

- AI-generated images
- Real images

## Model

The model uses transfer learning with ResNet50.

Training details:
- Backbone: ResNet50
- Dataset classes:
  - fake
  - real
- Loss function: CrossEntropyLoss
- Optimizer: Adam

## Usage

Upload an image and the model will predict whether it is AI-generated or real.