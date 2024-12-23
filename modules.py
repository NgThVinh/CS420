import cv2
import numpy as np
from tqdm import tqdm
import urllib

from model.ModelHandler import find_distance, find_threshold, DetectionModel, EmbeddingModel
from utils import timeit


@timeit(text='Time to load video: ')
def load_video_to_array(VIDEO_PATH):
    min_height, min_width = 256, 256
    original_height, original_width = None, None
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames = []
    while True:
        ret, frame = cap.read()   
        if ret == False:
            break
        if not original_height or not original_width:
            original_height, original_width = frame.shape[:2]
        h_ratio = min_height / original_height
        w_ratio = min_width / original_width
        resize_ratio = max(h_ratio, w_ratio)
        
        new_height = int(original_height * resize_ratio)
        new_width = int(original_width * resize_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        all_frames.append(resized_frame)
    return np.array(all_frames)

@timeit(text='Time to save video: ')
def save_video_from_array(video_array, output_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_array[0].shape[1], video_array[0].shape[0]))
    for frame in video_array:
        out.write(frame)
    out.release()

@timeit(text='Time to load image: ')
def load_image(path: str|list, is_url: bool=False):
    if isinstance(path, list):
        return [load_image(p, is_url) for p in path]
    if is_url:
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)
    else:
        return cv2.imread(path)

def check_distance(distances, threshold):
    confidence = np.average(distances)
    return confidence <= threshold

def __flatten_list__(nested_list):
    """
    Flatten a list of lists and capture the structure for restoring.
    Args:
        nested_list (list): A nested list where each sublist represents a set of items.
    Returns:
        np.array: Flattened list.
        list: Structure of the original nested list for restoration.
    """
    result = []
    structure = []
    for item in nested_list:
        structure.append(len(item))  # Record the length of each sublist
        if len(item) > 0:
            result.extend(item)  # Flatten items into result list
    return np.array(result), structure

def __restore_list__(flat_list, structure):
    """
    Restore a flat list to its original nested structure.
    Args:
        flat_list (np.array): Flattened array to restore.
        structure (list): List containing the lengths of each sublist.
    Returns:
        list: Restored list matching the original nested structure.
    """
    result = []
    index = 0
    for length in structure:
        if length == 0:
            result.append(np.array([]))  # Empty array for zero-length elements
        else:
            result.append(np.array(flat_list[index:index + length]))  # Slice and restore
            index += length  
    return result

def __get_batch_matched_map__(frames, query_embeddings, model_name, detector_backend, distance_metric, threshold):
    """
    Process a batch of frames to find matches based on embeddings and threshold criteria.
    Args:
        frames (list): List of frame images.
        query_embeddings (list): List of query embeddings for comparison.
        model_name (str): Model name for embedding extraction.
        detector_backend (str): Detector backend used for face detection.
        distance_metric (str): Metric for calculating distance between embeddings.
        threshold (float): Threshold for determining match.
    Returns:
        np.array: Array where each index indicates if a match was found for that frame.
    """
    matched_map = np.zeros(len(frames))  # Initialize match map
    list_faces = []
    
    # Load model client 
    embedding_model = EmbeddingModel(model_name)
    
    # Detect faces and preprocess for each frame
    detection_model = DetectionModel(detector_backend)
    for frame in frames:
        face_images = detection_model(frame, align=True)
        preprocess_images = np.array([embedding_model.preprocess_input(face_obj[0]) for face_obj in face_images])
        list_faces.append(preprocess_images)
    
    flatten, structure = __flatten_list__(list_faces)
    
    # If no faces found, return early
    if len(flatten) == 0:
        return matched_map
    
    # Get embeddings for detected faces
    embeddings = embedding_model.get_embedding(flatten, batch=True, preprocess_input=False)
    
    # Calculate distances and check threshold for each query embedding
    distances = []
    for query_embedding in query_embeddings:
        distances.append(np.array([find_distance(query_embedding, embedding, distance_metric) for embedding in embeddings]))
    
    # Apply threshold check and restore structure
    distances = np.apply_along_axis(lambda x: check_distance(x, threshold=threshold), axis=0, arr=distances)
    list_distances = __restore_list__(distances, structure)
    
    # Update matched_map based on max distances in restored list
    for idx, dist_list in enumerate(list_distances):
        if len(dist_list) > 0 and sum(dist_list):
            matched_map[idx] = 1
    
    return matched_map

@timeit(text='Time to detect people: ')
def get_matched_map(frames, query_images, model_name, detector_backend,
                    threshold=None, distance_metric='cosine', batch_size=128):
    """
    Process frames in batches to find matches with query images.
    Args:
        frames (list): List of frames to process.
        query_images (list): List of query images.
        model_name (str): Model name for embedding extraction.
        detector_backend (str): Detector backend for face detection.
        threshold (float): Threshold for determining match.
        distance_metric (str, optional): Metric to use for distance calculation. Defaults to 'cosine'
        batch_size (int, optional): Number of frames to process per batch. Defaults to 128.
    Returns:
        np.array: Array where each index indicates if a match was found for that frame.
    """
    num_batches = (len(frames) + batch_size - 1) // batch_size  # Calculate number of batches
    matched_map = np.zeros(len(frames))  # Initialize match map
    
    # Process and extract embeddings from query images
    q_embeddings = []
    embedding_model = EmbeddingModel(model_name)
    if embedding_model.model is None:
        print("Invalid model name. Choose one of:", embedding_model.__valid_name__)
        return matched_map
    
    # Detect faces and preprocess for each frame
    detection_model = DetectionModel(detector_backend)
    if detection_model.model is None:
        print("Invalid detector backend. Choose one of:", detection_model.__valid_name__)
        return matched_map

    for query in query_images:
        faces = detection_model(query, align=True)
        if len(faces) != 1:  # Ensure single face per query
            print("No face or more than 1 face detected in query image!")
            continue
        query_image, _ = faces[0]
        q_embedding = embedding_model.get_embedding(query_image)
        q_embeddings.append(q_embedding)
    if len(q_embeddings) == 0:
        print("No face or more than 1 face detected in query images!")
        return matched_map
    q_embeddings = np.array(q_embeddings)

    if not threshold:
        threshold = find_threshold(model_name, distance_metric)
    
    # Process frames in batches
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frames))
        
        batch_frames = frames[start_idx:end_idx]
        
        # Get batch matched map and combine results
        batch_matched_map = __get_batch_matched_map__(
            batch_frames,
            q_embeddings,
            model_name=model_name,
            detector_backend=detector_backend,
            threshold=threshold,
            distance_metric=distance_metric
        )
        
        matched_map[start_idx:end_idx] = batch_matched_map  # Update matched map with batch result
    
    return matched_map

# def preprocess_image(image, target_size=(160, 160)):
#     # image = cv2.resize(image, target_size)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = image / 255
#     return image
    
# def get_query_embedding_vector(detection_model, embedding_model, image) -> np.array:
#     prep_image = preprocess_image(image)
#     face_images = detection_model(prep_image)
#     if not face_images:
#         return np.array([])
#     elif len(face_images) > 1:
#         print("Your image has more than 1 person!")
#         return np.array([])
#     face_image, _ = face_images[0]
#     embedding_vector = embedding_model.get_embedding(face_image)
#     return embedding_vector

# def cosine_similarity(embedding_1, embedding_2):
#     embedding_1 = np.array(embedding_1).reshape(1, -1)
#     embedding_2 = np.array(embedding_2).reshape(1, -1)
#     cosine_similarity = np.dot(embedding_1, embedding_2.T) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
#     return cosine_similarity[0][0]

# def euclidean_similarity(embedding_1, embedding_2):
#     embedding_1 = np.array(embedding_1)
#     embedding_2 = np.array(embedding_2)
#     return np.linalg.norm(embedding_1 - embedding_2)

# @timeit(text='Time to detect people: ')
# def detect_people_from_frames(frames: list, query_image: np.array, 
#                               model_name: str = "ArcFace", threshold=None) -> list:
#     """
#     Detect faces from frames. Return matched map that has the same length as frames.
#     \nEach element is 1 if the frame is matched, 0 otherwise.
    
#     Args:
#         frames (list): list of frames
#         query_image (np.array): query image
#         model_name (str): model name (default: "custom_embedding_model")
#         threshold (float): threshold for similarity (default: None)
        
#         Returns:
#             list: matched map
#     """
#     detection_model = ModelHandler.DetectionModel("YOLO")
#     embeding_model = ModelHandler.EmbeddingModel(model_name)
#     if not threshold:
#         threshold = embeding_model.model.__threshold__

#     query_feature = get_query_embedding_vector(detection_model, embeding_model, query_image)
#     matched_map = np.zeros(len(frames))
#     for idx in tqdm(range(len(frames))):
#         prep_frame = preprocess_image(frames[idx])
#         face_images = detection_model(prep_frame)
#         if not face_images: # No face detected
#             continue
#         for face_image, _ in face_images:
#             embedding = embeding_model.get_embedding(face_image)
#             if cosine_similarity(query_feature, embedding) > threshold:
#                 matched_map[idx] = 1
#                 break
#     return matched_map
