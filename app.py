import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("steel_defect_clf.keras")

model = load_model()

# UI
st.title("Steel Defect Classifier")
st.write("Upload an image of steel surface to predict the defect type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.markdown(f"### Predicted: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
