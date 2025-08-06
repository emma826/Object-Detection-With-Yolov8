import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.title("🎥 YOLOv8 Object Detection By Amoke Emmanuel")

# Load model
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error("Model not found at: " + model_path)
    st.stop()

model = YOLO(model_path)

# Upload video
uploaded = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name
    st.video(input_path)

    # Select processing speed
    speed_option = st.selectbox(
        "⚙️ Choose processing speed:",
        ["Fast (skip 100 frames)", "Slow (skip 2 frames)"]
    )
    skip = 100 if "Fast" in speed_option else 2

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        st.stop()

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temp output video
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    output_path = tmp_out.name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')

    # Count and buffer frames
    saved_frames = []
    count = 0
    st.write("🔄 Processing...")
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % skip != 0:
            continue
        results = model(frame)
        annotated = results[0].plot()
        saved_frames.append(annotated)
        progress.progress(min(count / total_frames, 1.0))

    cap.release()
    os.unlink(input_path)

    # Adjust FPS to maintain real-time playback duration
    frames_saved = len(saved_frames)
    output_fps = (frames_saved / total_frames) * fps
    output_fps = max(output_fps, 1)  # Prevent 0 FPS

    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    for frame in saved_frames:
        out.write(frame)
    out.release()

    st.success("✅ Video processed successfully!")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Processed Video", f.read(), "output.webm", "video/webm")
