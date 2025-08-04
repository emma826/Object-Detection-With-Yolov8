import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("Object Detection on Video using YOLOv8 By Amoke Emmanuel")
st.write("Upload a video file to perform object detection.")

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)

    return model

model_path = '/content/runs/detect/yolov8n_custom_model_trained/weights/best.pt'

try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if video_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    st.subheader("Processing Video...")
    video_capture = cv2.VideoCapture("temp_video.mp4")
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total frames: {frame_count}")

    video_placeholder = st.empty()

    for frame_idx in range(frame_count):
        ret, frame = video_capture.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            im_array = r.plot()
            frame_with_detections = Image.fromarray(im_array[..., ::-1])

        video_placeholder.image(frame_with_detections, channels="RGB", use_column_width=True)

    video_capture.release()
    st.success("Video processing complete.")

    import os
    os.remove("temp_video.mp4")

else:
    st.info("Please upload a video file to start detection.")