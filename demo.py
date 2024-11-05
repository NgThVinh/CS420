import cv2
import numpy as np
from modules import load_video_to_array, save_video_from_array, load_image, get_matched_map
from utils import remove_duplicate

QUERY_IMAGE_PATH = ['data/target_in.png', 'data/target_out1.png', 'data/target_out2.png']
VIDEO_PATH = 'data/mv_oldtownroad.mp4'


def main():
    print("Loading Images...")
    query_images = load_image(QUERY_IMAGE_PATH)
    print("Image shape:", [query_image.shape for query_image in query_images])

    print("Loading Video...")
    video_array = load_video_to_array(VIDEO_PATH)
    print("Total Frame:", video_array.shape)

    print("Optimizing data...")
    method = "PHash"
    unique_frames, duplicates_map = remove_duplicate(video_array, method)
    print(f"Removed duplication using {method}. \nUnique Frame:", len(unique_frames))

    embedding_model = "ArcFace"
    detection_model = "retinaface"
    distance_metric = "cosine"
    print(f"Detecting faces using {embedding_model} and {detection_model} with {distance_metric} distance metric...")
    matched_map = get_matched_map(list(unique_frames.values()), 
                                    query_images, 
                                    model_name=embedding_model,
                                    detector_backend=detection_model,
                                    threshold=None,
                                    distance_metric=distance_metric,
                                    batch_size=128
                                    )
    print("Matched frames:", matched_map[matched_map == 1].shape[0])

    # Reconstruct matched frame map to match the original video array
    matched_frame = np.zeros(len(video_array))
    for f_idx, flag in zip(unique_frames.keys(), matched_map):
        if not flag:
            continue
        matched_frame[f_idx] = 1
        for dup_frame in duplicates_map[str(f_idx)]:
            matched_frame[int(dup_frame)] = 1
    print("Reconstructed map to match original video. \nTotal matched frames:", 
          matched_frame[matched_frame == 1].shape[0])
    
    print("Creating video...") 
    output_path = 'data/output.avi'
    output_video = []
    for idx, frame in enumerate(video_array):
        if matched_frame[idx] == 1:
            frame = cv2.putText(frame, "Matched", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 
                                cv2.LINE_AA)
        output_video.append(frame)
    output_video = np.array(output_video)
    print("Output video shape:", output_video.shape)

    print("Saving video...")
    save_video_from_array(output_video, output_path)
    print("Output video saved at:", output_path)

if __name__ == '__main__':
    main()
