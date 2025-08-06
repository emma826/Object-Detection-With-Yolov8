import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import tempfile
import os

st.title('Object Detection with YOLOv8')
st.write('Upload a video to detect objects using your YOLOv8 model.')

# Load your trained model
model_path = 'best.pt'  # Change if needed
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            input_video_path = tmp_input.name

        st.video(input_video_path)  # Show original video

        st.write("🔄 Processing video, please wait...")

        # Prepare video reader/writer
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            skip_interval = 5
            frame_count = 0
            processed_frames = 0

            # Output video file
            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video_path = tmp_output.name

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            status = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_interval != 0:
                    continue

                results = model(frame)
                annotated = results[0].plot()
                out.write(annotated)

                processed_frames += 1
                status.write(f"✅ Processed {frame_count} / {total_frames} frames")
                progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            os.unlink(input_video_path)

            st.success("🎉 Video processing complete!")
            st.video(output_video_path)

            # Download button
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="📥 Download Result Video",
                    data=f,
                    file_name="detected_output.mp4",
                    mime="video/mp4"
                )
