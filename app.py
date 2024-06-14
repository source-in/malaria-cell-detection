import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = tf.keras.models.load_model('malaria_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img = np.asarray(img)
    img = img / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class of the image
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Malaria Cell Detection")

st.write("Upload an image of a cell to predict if it is infected with malaria.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image)

    if prediction[0][0] > prediction[0][1]:
        st.write("The cell is Parasitized.")
    else:
        st.write("The cell is Uninfected.")
