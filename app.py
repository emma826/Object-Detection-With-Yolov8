import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import tempfile
import os
from io import BytesIO

st.title('🎯 Object Detection with YOLOv8')
st.write('Upload a video and detect objects using your trained YOLOv8 model.')

model_path = 'best.pt'  # ✅ Replace with your correct model path

if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
else:
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.getvalue())
            input_video_path = tmp_in.name

        st.video(input_video_path)  # Show input video

        st.write("🚀 Processing video...")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file.")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            skip_interval = 10
            frame_count = 0

            # Output path
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video_path = tmp_out.name

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            status = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % skip_interval != 0:
                    continue

                results = model(frame)
                annotated = results[0].plot()
                out.write(annotated)

                progress.progress(min(frame_count / total_frames, 1.0))
                status.text(f"Processing frame {frame_count}/{int(total_frames)}...")

            cap.release()
            out.release()
            os.unlink(input_video_path)

            st.success("✅ Processing complete!")

            # Make sure file is fully written before loading
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)  # ✅ Stream the annotated video
            st.download_button(
                "📥 Download processed video",
                video_bytes,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
