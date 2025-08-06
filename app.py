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

        # Create a temporary file for the output video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp_file:
            output_video_path = out_tmp_file.name

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            os.unlink(video_path) # Clean up temporary input file
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for .mp4 files
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            st.write("Running object detection on video frames...")

            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform object detection
                results = model(frame)

                # Plot results on the frame
                annotated_frame = results[0].plot()

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                frame_count += 1
                progress_bar.progress(frame_count / total_frames)


            # Release everything when job is finished
            cap.release()
            out.release()

            st.write("Finished processing video.")

            # Display the processed video
            st.video(output_video_path)

            # Clean up temporary files
            os.unlink(video_path)
            os.unlink(output_video_path)