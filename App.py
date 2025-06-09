import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Set page title and icon
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")

# Load the pre-trained model
@st.cache_resource
def load_my_model():
    return load_model('Hand Written Digits Recognition Using CNN.ipynb')

model = load_my_model()

# Title and description
st.title("Handwritten Digit Recognition using CNN")
st.write("""
Upload an image of a handwritten digit (0-9) or draw one in the canvas below, 
and the model will predict the digit.
""")

# Create two columns for upload and canvas
col1, col2 = st.columns(2)

with col1:
    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

with col2:
    # Drawing canvas
    st.write("Or draw a digit here:")
    canvas_result = st.canvas_drawing(
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas"
    )

# Process the image and make prediction
def predict_digit(img):
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255.0
    # Make prediction
    prediction = model.predict(img)
    return np.argmax(prediction), max(prediction[0])

# Display results
if uploaded_file is not None or canvas_result.image_data is not None:
    col1, col2, col3 = st.columns(3)
    
    if uploaded_file is not None:
        # Process uploaded file
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        with col1:
            st.image(image, caption="Uploaded Image", width=150)
    else:
        # Process canvas drawing
        img_array = np.array(canvas_result.image_data)
        # Convert RGBA to grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        with col1:
            st.image(img_array, caption="Your Drawing", width=150)
    
    # Make prediction
    digit, confidence = predict_digit(img_array)
    
    with col2:
        st.subheader("Prediction")
        st.write(f"Digit: **{digit}**")
        st.write(f"Confidence: {confidence*100:.2f}%")
    
    with col3:
        st.subheader("Image Processed")
        # Show the processed image (28x28)
        processed_img = cv2.resize(img_array, (28, 28))
        st.image(processed_img, width=100)
else:
    st.info("Please upload an image or draw a digit to get a prediction.")

# Add some information about the model
st.sidebar.header("About")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset 
to recognize handwritten digits (0-9).

Model architecture:
- 2 Conv2D layers with MaxPooling
- 2 Dense layers with Dropout
- Trained for 10 epochs
- Test accuracy: ~61%
""")
