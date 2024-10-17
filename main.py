import cv2
from func import remove_duplicate, load_video_to_array, load_image, detect_faces_from_frames

QUERY_IMAGE_PATH = 'data/target_out1.png'
VIDEO_PATH = 'data/mv_oldtownroad.mp4'

def main():
    print("Loading data...")
    q_image = load_image(QUERY_IMAGE_PATH)
    print("Image shape:", q_image.shape)
    video_array = load_video_to_array(VIDEO_PATH)
    print("Total Frame:", len(video_array))

    print("Processing data...")
    unique_frames, duplicates = remove_duplicate(video_array)
    print("Removed duplication! Unique Frame:", len(unique_frames))

    print("Detecting faces...")
    model_name = "deepface.DeepFace"
    matched_images = detect_faces_from_frames(model_name, unique_frames, q_image)

    matched_idx = []
    for idx, _ in matched_images:
        matched_idx.append(idx)
        if str(idx) in duplicates.keys():
            matched_idx.extend(int(f) for f in duplicates[str(idx)])
    matched_idx = sorted(matched_idx)
    print("Matched frames:", len(matched_idx))

    print("Displaying matched frames...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx in matched_idx:
            frame = cv2.putText(frame, "Matched", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("Window", frame)
        cv2.waitKey(0)

        idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
