import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # Changed this
import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.saving import register_keras_serializable

# --- 1. MODEL LOADING ---


def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model
# --- 2. STABILIZED VIDEO CLASS ---


model = create_model()
model.load_weights("facialemotionmodel.h5")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# --- 3. UI ---
st.title("Live Emotion Detector")

webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionTransformer,  # Changed factory name
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}
)
