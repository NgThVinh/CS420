import cv2
import numpy as np
from tqdm import tqdm
import urllib

from model import ModelHandler
from utils import timeit


@timeit(text='Time to load video: ')
def load_video_to_array(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames = []
    while True:
        ret, frame = cap.read()   
        if ret == False:
            break
        all_frames.append(frame)
    return np.array(all_frames)

@timeit(text='Time to save video: ')
def save_video_from_array(video_array, output_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_array[0].shape[1], video_array[0].shape[0]))
    for frame in video_array:
        out.write(frame)
    out.release()

@timeit(text='Time to load image: ')
def load_image(path, is_url=False):
    if is_url:
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)
    else:
        return cv2.imread(path)

def preprocess_image(image, target_size=(160, 160)):
    # image = cv2.resize(image, target_size)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image / 255
    return image
    
def get_query_embedding_vector(detection_model, embedding_model, image) -> np.array:
    prep_image = preprocess_image(image)
    face_images = detection_model(prep_image)
    if not face_images:
        return np.array([])
    elif len(face_images) > 1:
        print("Your image has more than 1 person!")
        return np.array([])
    face_image, _ = face_images[0]
    embedding_vector = embedding_model.get_embedding(face_image)
    return embedding_vector

def cosine_similarity(embedding_1, embedding_2):
    embedding_1 = np.array(embedding_1).reshape(1, -1)
    embedding_2 = np.array(embedding_2).reshape(1, -1)
    cosine_similarity = np.dot(embedding_1, embedding_2.T) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
    return cosine_similarity[0][0]

def euclidean_similarity(embedding_1, embedding_2):
    embedding_1 = np.array(embedding_1)
    embedding_2 = np.array(embedding_2)
    return np.linalg.norm(embedding_1 - embedding_2)

@timeit(text='Time to detect people: ')
def detect_people_from_frames(frames: list, query_image: np.array, 
                              model_name: str = "ArcFace", threshold=None) -> list:
    """
    Detect faces from frames. Return matched map that has the same length as frames.
    \nEach element is 1 if the frame is matched, 0 otherwise.
    
    Args:
        frames (list): list of frames
        query_image (np.array): query image
        model_name (str): model name (default: "custom_embedding_model")
        threshold (float): threshold for similarity (default: None)
        
        Returns:
            list: matched map
    """
    detection_model = ModelHandler.DetectionModel("YOLO")
    embeding_model = ModelHandler.EmbeddingModel(model_name)
    if not threshold:
        threshold = embeding_model.model.__threshold__

    query_feature = get_query_embedding_vector(detection_model, embeding_model, query_image)
    matched_map = np.zeros(len(frames))
    for idx in tqdm(range(len(frames))):
        prep_frame = preprocess_image(frames[idx])
        face_images = detection_model(prep_frame)
        if not face_images: # No face detected
            continue
        for face_image, _ in face_images:
            embedding = embeding_model.get_embedding(face_image)
            if cosine_similarity(query_feature, embedding) > threshold:
                matched_map[idx] = 1
                break
    return matched_map
