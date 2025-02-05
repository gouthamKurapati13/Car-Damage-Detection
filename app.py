import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# Load the trained model
model_path = "best.pt"
model = YOLO(model_path)

# Streamlit UI
st.title("🚗 Car Damage Detection - Techkrate - Goutham")
st.write("Upload an image to detect damages.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    results = model(image)

    # Display results
    for result in results:
        result.show()  # Display in notebook
        result.save("output.jpg")  # Save output image

    st.image("output.jpg", caption="Detected Damage", use_column_width=True)
