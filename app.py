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
st.write('Upload a video or image to detect objects using a custom-trained YOLOv8 model.')

# --- Frame Skipping Configuration (for videos) ---
# Set the frame skip interval. 1 means process every frame, 2 means process every 2nd frame, etc.
FRAME_SKIP_INTERVAL = 5 # You can adjust this value
# ------------------------------------


# Load the trained YOLOv8 model
# Ensure the model path is correct based on where you saved it
model_path = 'best.pt'  # Update this path if needed
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is downloaded to this location.")
else:
    # Load the model and ensure it's on the correct device (GPU if available)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu' # Keep this if you might run on GPU later
    device = 'cpu' # Explicitly setting to cpu for demonstration based on user feedback
    model = YOLO(model_path).to(device)
    st.write(f"Using device: {device}")


    uploaded_file = st.file_uploader("Choose a video or image file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type.startswith('video/'):
            # --- Video Processing ---
            st.write("Video file uploaded. Starting processing...")
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            st.write(f"Temporary video saved at: {video_path}")


            st.video(video_path)

            st.write("Processing video...")

            # Create a temporary file for the output video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp_file:
                output_video_path = out_tmp_file.name
            st.write(f"Output video will be saved to: {output_video_path}")

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                os.unlink(video_path) # Clean up temporary input file
            else:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                st.write(f"Video properties: Resolution={frame_width}x{frame_height}, FPS={fps}")


                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for .mp4 files
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                if not out.isOpened():
                    st.error("Error: Could not create video writer.")
                    cap.release()
                    os.unlink(video_path)
                    os.unlink(output_video_path)
                else:
                    st.write(f"Running object detection on video frames with a skip interval of {FRAME_SKIP_INTERVAL}...")

                    progress_bar = st.progress(0)
                    frame_count = 0
                    processed_frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.write(f"Total frames in video: {total_frames}")


                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.write("End of video or error reading frame.")
                            break

                        # Process frame only if it's a multiple of the skip interval
                        if frame_count % FRAME_SKIP_INTERVAL == 0:
                            # Perform object detection, explicitly using the determined device
                            # results = model(frame, device=device) # Use this if you uncommented GPU device
                            results = model(frame, device='cpu') # Explicitly using cpu based on user feedback

                            # Plot results on the frame
                            annotated_frame = results[0].plot()

                            # Write the annotated frame to the output video
                            out.write(annotated_frame)

                            processed_frame_count += 1
                            progress_bar.progress(frame_count / total_frames) # Progress based on total frames

                        frame_count += 1
                        # st.write(f"Processed frame {frame_count}") # Uncomment for very detailed frame-by-frame logging


                    # Release everything when job is finished
                    cap.release()
                    out.release()

                    st.write(f"Finished processing video. Processed {processed_frame_count} frames out of {total_frames}.")

                    # Display the processed video
                    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                        st.video(output_video_path)
                        st.write("Processed video displayed below.")
                    else:
                        st.error("Output video file was not created or is empty.")


                    # Clean up temporary files
                    st.write(f"Cleaning up temporary files: {video_path}, {output_video_path}")
                    os.unlink(video_path)
                    os.unlink(output_video_path)


        elif file_type.startswith('image/'):
            # --- Image Processing ---
            st.write("Processing image...")

            # Read the image using PIL
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image) # Convert to numpy array for OpenCV/YOLO

            st.image(image, caption="Original Image", use_container_width=True)

            st.write("Running object detection on image...")
            # Perform object detection
            # results = model(image_np, device=device) # Use this if you uncommented GPU device
            results = model(image_np, device='cpu') # Explicitly using cpu based on user feedback


            # Plot results on the image
            annotated_image = results[0].plot() # results[0].plot() returns a numpy array (BGR format)

            # Convert the annotated image (numpy array BGR) to RGB for Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            st.image(annotated_image_rgb, caption="Image with Object Detection", use_container_width=True)
            st.write("Finished processing image.")

        else:
            st.error("Unsupported file type.")
            st.write(f"Uploaded file type: {file_type}")

    else:
        st.write("Please upload a video or image file to begin.")