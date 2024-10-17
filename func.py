import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from imagededup.methods import PHash

import urllib
from ultralytics import YOLO
from deepface import DeepFace

EMBEDDING_MODEL_PATH = 'model/CustomEmbedingModel.keras'
YOLO_MODEL_PATH = 'model/yolov8n-face.pt'
deepface_model = 'ArcFace' # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet


def remove_duplicate(frames, max_distance_threshold=10):
    phasher = PHash()
    encodings = {}
    unique_frames = {}

    for frame_idx, frame in enumerate(frames):
        unique_frames.update({frame_idx: frame})
        encodings.update({str(frame_idx): phasher.encode_image(image_array=frame)})
        frame_idx+=1
    
    duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=max_distance_threshold)
    
    for frame_idx, dup_frames in duplicates.items():
        if int(frame_idx) in unique_frames.keys():
            try:
                for dup_frame in dup_frames:
                    unique_frames.pop(int(dup_frame)) 
            except:
                pass
        
    return unique_frames, duplicates

def load_video_to_array(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []

    for frame_idx in range(n_frames):
        ret, frame = cap.read()   
        if ret == False:
            break
        all_frames.append(frame)
    return all_frames

def load_image(path, is_url=False):
    if is_url:
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)
    else:
        return cv2.imread(path)
    
class CustomEmbedding:
    def __init__(self, PATH):
        self.__name__ = "custom_embedding_model"
        self.model = self.get_embedding_model(PATH)
        self.INPUT_SHAPE = self.model.input.shape[1:]

    def get_embedding_model(self, PATH):
        def triplet_loss(margin=0.5):
            def loss(y_true, y_pred):
                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

                pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
                neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
                # triplet loss
                loss_value = tf.maximum(pos_dist - neg_dist + margin, 0.0)
                return tf.reduce_mean(loss_value)
            return loss

        def l2_normalize(x):
            return tf.math.l2_normalize(x, axis=1)

        custom_objects = {
            "triplet_loss": triplet_loss,
            'l2_normalize': l2_normalize,
        }

        embedding_model = tf.keras.models.load_model(PATH,
                                                     custom_objects=custom_objects,
                                                     safe_mode=False)
        embedding_model.__name__ = "custom_embedding_model"
        return embedding_model
    
    def preprocess_image(self, face):
        INPUT_SHAPE = self.INPUT_SHAPE
        input_img = cv2.resize(face, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        return input_img / 255
    
    def predict_similarity(self, img1, img2):
        embedding_1 = self.model.predict(np.expand_dims(img1, axis=0),verbose=False)
        embedding_2 = self.model.predict(np.expand_dims(img2, axis=0),verbose=False)
        cosine_similarity = np.dot(embedding_1, embedding_2.T) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
        return cosine_similarity[0][0]

    def feat_extract(self, img):
        return self.model.predict(np.expand_dims(img, axis=0),verbose=False)

    def cosine_similarity(self, embedding_1, embedding_2):
        cosine_similarity = np.dot(embedding_1, embedding_2.T) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
        return cosine_similarity[0][0]

def detect_faces_from_image(img):
    # img = cv2.resize(img, (640, 640))
    yolo_model = YOLO(YOLO_MODEL_PATH, verbose=False)
    results = yolo_model(img, verbose=False)
    list_faces = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            face_img = img[y1:y2, x1:x2]
            list_faces.append((face_img, conf))
    return list_faces
    
def get_image_embedding(model, query_image):
    face_images = detect_faces_from_image(query_image)
    if not face_images:
        return []
    elif len(face_images) > 1:
        print("Your image has more than 1 person!")
        return []
    face_image, conf = face_images[0]
    if model.__name__ == "custom_embedding_model":
        prep_image = model.preprocess_image(face_image)
        query_feature = model.feat_extract(prep_image)
        
        return query_feature
    
    elif model.__name__ == 'deepface.DeepFace':
        embedding = model.represent(face_image, 
                                    model_name=deepface_model,
                                    detector_backend='skip')[0]['embedding']
        return embedding
    
    else:
        print(f"invalid model {model.__name__}")
        return
        
def verify(model, query_embedding, image, THRESHOLD=0.5):
    if model.__name__ == "custom_embedding_model":
        prep_image = model.preprocess_image(image)
        embedding = model.feat_extract(prep_image)
        return model.cosine_similarity(query_embedding, embedding) > THRESHOLD
    
    elif model.__name__ == 'deepface.DeepFace':
        result = model.verify(
            img1_path = query_embedding,
            img2_path = image,
            model_name=deepface_model,
            detector_backend='skip',
            enforce_detection=False,
            silent=True
        )
        return result['verified']
    else:
        print(f"invalid model {model.__name__}")
        return False
        
def detect_faces_from_frames(model_name, frames, query_image) -> list:
    if model_name == "custom_embedding_model":
        model = CustomEmbedding(EMBEDDING_MODEL_PATH)
    elif model_name == "deepface.DeepFace":
        model = DeepFace
    else:
        print("invalid model")
        return []
    
    query_feature = get_image_embedding(model, query_image)
    
    matched_img = []
    for frame_idx, frame in tqdm(frames.items()):
        face_images = detect_faces_from_image(frame)
        if not face_images:
            continue
        for face_image, conf in face_images:
            if verify(model, query_feature, face_image):
                matched_img.append([frame_idx, frame])
                break
                
    return matched_img