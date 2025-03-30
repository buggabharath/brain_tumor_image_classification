import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
@st.cache_resource  # Cache model loading for efficiency
def load_model():
    model = tf.keras.models.load_model("ResNet50V2_model.h5")  # Replace with your model file
    return model

# Define class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']  # Replace with your class names

# Prediction function
def predict_image(model, image, image_size=(224, 224)):
    """
    Preprocesses the image and makes a prediction.
    """
    # Resize and preprocess image
    image = image.resize(image_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    y_prob = model.predict(img_array)
    y_pred = np.argmax(y_prob, axis=1)[0]  # Get class index
    confidence = np.max(y_prob) * 100  # Get confidence score

    return class_names[y_pred], confidence, y_prob

# Streamlit UI
st.title("Brain Tumour Image Classification App")
st.write("Upload an image, and the model will predict the class.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_model()

    # Make prediction
    predicted_label, confidence, probabilities = predict_image(model, image)

    # Show result
    st.subheader("🔹 Prediction:")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show probability distribution
    st.bar_chart({class_names[i]: probabilities[0][i] for i in range(len(class_names))})

st.write("Brain Tumour Detection...")
