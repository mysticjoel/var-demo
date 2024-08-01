import streamlit as st
import warnings
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings("ignore")

# Inject custom CSS to change color and set background image
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://wallpapercave.com/wp/wp9006032.jpg');
        background-size: cover;
    }
    .st-bc {
        background-color: #47ea4e !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_file_path(uploaded_file):
    # Save the uploaded file to the temporary directory
    file_path = "./temp/clip"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    logging.info(f"File saved to: {file_path}")
    return file_path

@st.cache_resource
def model_loader(model_path):
    model = keras.models.load_model(model_path)
    logging.info(f"Model loaded from: {model_path}")
    return model

def pred_func1(video_path, model):
    ans = 0
    logs = []
    result = ""
    capture_video = cv2.VideoCapture(video_path)

    if not capture_video.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return "Error", 0, []

    while True:
        (taken, frame) = capture_video.read()
        if not taken:
            break
        logging.info(f"Frame shape: {frame.shape}")
        frame = cv2.resize(frame, (256, 256)).astype("float32")
        preds = model.predict(np.expand_dims(frame, axis=0), verbose=0)[0]
        logging.info(f"Predictions: {preds}")
        logs.append(preds)
        ans = max(ans, preds)
        
    capture_video.release()
    
    result = "Foul" if ans > 1e-25 else "Clean"
    logs = np.array(logs).ravel()
    return result, ans, logs

def pred_func2(video_path, model):
    ans = 0
    logs = []
    result = ""
    capture_video = cv2.VideoCapture(video_path)

    if not capture_video.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return "Error", 0, []

    while True:
        (taken, frame) = capture_video.read()
        if not taken:
            break
        logging.info(f"Frame shape: {frame.shape}")
        frame = cv2.resize(frame, (256, 256)).astype("float32")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=2)
        preds = model.predict(np.expand_dims(gray, axis=0), verbose=0)[0]
        logging.info(f"Predictions: {preds}")
        logs.append(preds)
        ans = max(ans, preds)

    capture_video.release()

    result = "Foul" if ans >= 0.29 else "Clean"
    logs = np.array(logs).ravel()
    return result, ans, logs

def main():
    st.title("AI Referee Assistant")
    st.subheader("Player Contact and Foul Detection using a CNN-based machine learning model.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "3gp"])
    
    st.sidebar.title("Models")
    model_version = st.sidebar.radio("Select Model", ("V1 (RGB Rendering)", "V2 (Grayscale Rendering)"))

    if uploaded_file is not None:
        file_path = get_file_path(uploaded_file)
        st.video(uploaded_file)

        if model_version == "V1 (RGB Rendering)":
            st.subheader("V1: RGB Rendering")
            model = model_loader('./model_1.h5')
            result, ans, logs = pred_func1(file_path, model)
            st.subheader(f"Prediction: {result}")
        else:
            st.subheader("V2: Grayscale Rendering")
            model = model_loader('./model_2.h5')
            result, ans, logs = pred_func2(file_path, model)
            st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()
