# Đồ án CS420

# Features
- Face Matching: Uses embedding and detection models to find matching faces in a video.
- Duplicate Frame Removal: Optimizes the input video by removing duplicate frames based on a hashing method.
- Customizable Parameters: Allows customization of embedding models, detection models, threshold, and distance metrics.
- Output Video: Generates a video with frames containing matches annotated with a "Matched" label.

# Requirements
Python 3.10

```
pip install -r requirements.txt
```

# Example Command

```
python demo.py --query_image_paths path/to/image1.jpg path/to/image2.jpg \
    --video_path path/to/video.mp4 \
    --embedding_model ArcFace \
    --detection_model retinaface \
    --hash_method PHash \
    --threshold 0.6 \
    --distance_metric cosine \
    --batch_size 128
```
