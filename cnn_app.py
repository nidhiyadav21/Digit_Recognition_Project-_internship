import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Load the CNN model
model = load_model("mnist_cnn_model.keras",compile=False)


# Title
st.title("🧠 CNN-based Handwritten Digit Classifier")

# Choose input method
option = st.radio("Choose Input Method", ("Upload Image", "Draw Digit"))

# Image uploader
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        st.image(image, caption="Processed Image", width=150)

        img_array = np.array(image) / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)

        if st.button("Predict"):
            prediction = np.argmax(model.predict(img_input))
            st.success(f"🎯 Predicted Digit: {prediction}")

# Draw section
elif option == "Draw Digit":
    st.write("Draw a digit (0–9) below:")

    canvas = st_canvas(
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        image = Image.fromarray(canvas.image_data).convert("L")
        image = image.resize((28, 28))
        st.image(image, caption="Processed Drawing", width=150)

        img_array = np.array(image) / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)

        if st.button("Predict"):
            prediction = np.argmax(model.predict(img_input))
            st.success(f"🎯 Predicted Digit: {prediction}")
