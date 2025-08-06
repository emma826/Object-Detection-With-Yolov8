import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.title("🎥 YOLOv8 Object Detection By Amoke Emmmanuel")
st.write('Upload a video to detect objects using a custom-trained YOLOv8 model.')

model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error("Model file not found.")
else:
    model = YOLO(model_path)
    uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded.read())
            in_path = tmp_in.name
        st.video(in_path)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            st.error("Failed to open.")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            out_path = tmp_out.name
            # Use VP80 codec for webm
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            count = 0
            skip = 2

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                if count % skip != 0:
                    continue
                res = model(frame)
                annotated = res[0].plot()
                out.write(annotated)
                progress.progress(min(count/total, 1.0))

            cap.release()
            out.release()
            os.unlink(in_path)

            st.success("Processing complete!")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("📥 Download Video", f.read(), "output.webm", "video/webm")
