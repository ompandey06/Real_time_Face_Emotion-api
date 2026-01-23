import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.saving import register_keras_serializable

# --- 1. MODEL LOADING LOGIC (CRITICAL) ---


@register_keras_serializable()
class CustomSequential(Sequential):
    pass


# This must be outside the class so it loads once when the app starts
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json, custom_objects={
                        "Sequential": CustomSequential})
model.load_weights("facialemotionmodel.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --- 2. VIDEO PROCESSING LOGIC ---


class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            label = EMOTIONS[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 3. FRONTEND UI ---
st.set_page_config(page_title="Emotion Detector")
st.title("Real-Time Face Emotion Detector")
st.write("Click 'Start' below to begin.")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    # This configuration helps the browser find your webcam
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
