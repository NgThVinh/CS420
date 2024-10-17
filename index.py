import streamlit as st
import cv2
import os
import tempfile
from PIL import Image

# Load your human detection model here (e.g., a pre-trained model)
# For demo purposes, this function just returns True for every frame.
def detect_human_in_frame(frame):
    # Apply your model to detect humans in the frame
    return True  # Example logic for detecting human presence in the frame

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        if detect_human_in_frame(frame):
            extracted_frames.append(frame_path)
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1

    cap.release()
    return extracted_frames

def main():
    st.title("Video and Image Upload for Human Detection")

    # Upload video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    # Upload image set
    uploaded_images = st.file_uploader("Upload a set of images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_video and uploaded_images:
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            temp_video_path = temp_video_file.name
        
        # Create a temporary folder for extracted frames
        temp_folder = tempfile.mkdtemp()

        # Extract frames with human detection
        st.write("Extracting frames with humans detected...")
        extracted_frames = extract_frames(temp_video_path, temp_folder)

        # Display extracted frames
        st.write(f"Found {len(extracted_frames)} frames with humans:")
        for frame_path in extracted_frames:
            st.image(Image.open(frame_path), caption=frame_path)

if __name__ == "__main__":
    main()
