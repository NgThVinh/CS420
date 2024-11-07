import cv2
import numpy as np
from modules import load_video_to_array, save_video_from_array, load_image, get_matched_map
from utils import remove_duplicate
import argparse

def main(query_image_paths, video_path, embedding_model, detection_model, 
         hash_method = "PHash", threshold=None, distance_metric="cosine", batch_size=128):
    
    print("Loading Images...")
    query_images = load_image(query_image_paths)
    print("Image shape:", [query_image.shape for query_image in query_images])

    print("Loading Video...")
    video_array = load_video_to_array(video_path)
    print("Total Frame:", video_array.shape)

    print("Optimizing data...")
    
    unique_frames, duplicates_map = remove_duplicate(video_array, hash_method)
    print(f"Removed duplication using {hash_method}. \nUnique Frame:", len(unique_frames))

    print(f"Detecting faces using {embedding_model} and {detection_model} with {distance_metric} distance metric...")
    matched_map = get_matched_map(
        list(unique_frames.values()), 
        query_images, 
        model_name=embedding_model,
        detector_backend=detection_model,
        threshold=threshold,
        distance_metric=distance_metric,
        batch_size=batch_size
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
    parser = argparse.ArgumentParser(description="Video Face Detection Script")
    parser.add_argument("--query_image_paths", nargs="+", required=True, help="Paths to the query images")
    parser.add_argument("--video_path", required=True, help="Path to the video file")
    parser.add_argument("--embedding_model", default="ArcFace", help="Embedding model name")
    parser.add_argument("--detection_model", default="retinaface", help="Detection model name")
    parser.add_argument("--hash_method", default="PHash", help="Hash method for removing duplication")
    parser.add_argument("--threshold", type=float, default=None, help="Matching threshold")
    parser.add_argument("--distance_metric", default="cosine", help="Distance metric (cosine, euclidean, etc.)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing frames")

    args = parser.parse_args()
    main(args.query_image_paths, args.video_path, args.embedding_model, args.detection_model, args.hash_method, 
         args.threshold, args.distance_metric, args.batch_size)
