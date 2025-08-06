import streamlit as st
import tempfile

st.title("Object Detection in Video By Amoke Emmanuel")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    st.write("Running object detection...")

    # Load the custom trained model
    model = YOLO('/content/runs/detect/yolov8n_custom_model_trained/weights/best.pt') # Make sure to use the correct path to your trained model weights

    # Run inference on the video
    results = model(video_path, stream=True)

    
    st.write("Detection results:")
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im_array = im_array[..., ::-1]  # convert BGR to RGB
        st.image(im_array, caption="Detected Objects", use_column_width=True)

    os.unlink(video_path) # Clean up the temporary file