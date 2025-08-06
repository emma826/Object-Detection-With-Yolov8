import streamlit as st
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import tempfile
import time
import os

st.title('Object Detection with YOLOv8')
st.write('Upload a video to detect objects using a custom-trained YOLOv8 model.')

# Load the trained YOLOv8 model
model_path = 'best.pt'  # Update this if needed
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        st.video(video_path)  # Show the original video

        st.write("Processing video...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            stframe = st.empty()  # Placeholder for frames
            skip_interval = 5     # Process every 5th frame
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_interval != 0:
                    continue

                # Inference
                results = model(frame)
                annotated_frame = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display annotated frame
                stframe.image(annotated_rgb, channels="RGB", use_column_width=True)

                time.sleep(0.05)  # Delay to allow rendering (adjust if needed)

            cap.release()
            os.unlink(video_path)
            st.success("Finished processing video.")
