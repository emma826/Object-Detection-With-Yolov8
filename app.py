import streamlit as st
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import tempfile

st.title('Object Detection with YOLOv8 by Amoke Emmanuel')
st.write('Upload a video to detect objects using a custom-trained YOLOv8 model.')

# Load the trained YOLOv8 model
# Ensure the model path is correct based on where you saved it
model_path = 'best.pt'  # Update this path if needed
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is downloaded to this location.")
else:
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        st.video(video_path)

        st.write("Processing video...")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Create a placeholder to display the video with detections
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform object detection
                results = model(frame)

                # Plot results on the frame
                annotated_frame = results[0].plot()

                # Convert the annotated frame from BGR to RGB for displaying in Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the annotated frame
                stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            os.unlink(video_path) # Delete the temporary file
            st.write("Finished processing video.")