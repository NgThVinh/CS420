import numpy as np
from ultralytics import YOLO
from model import CustomEmbedding
from deepface import DeepFace

YOLO_MODEL_PATH = 'model/yolov8n-face.pt'

class DetectionModel:
    def __init__(self, name: str):
        self.__valid_name__ = ["YOLO"]
        self.get_model(name)
        
    def get_model(self, model_name) -> None:
        if model_name not in self.__valid_name__:
            print(f"Invalid model name: {model_name}")
            return 
        if model_name == "YOLO":
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model.__name__ = model_name
    
    def __call__(self, image, verbose=False) -> list:
        """
        Args:

        image: np.array - image to detect faces
        verbose: bool (default=False) - whether to print the result or not

        Returns:
        
        list of tuple(face_img, confidence) - list of detected faces with confidence
        """
        if not self.model:
            return None
        if self.model.__name__ == "YOLO":
            results = self.model(image, verbose=verbose)
            list_faces = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    face_img = image[y1:y2, x1:x2]
                    list_faces.append((face_img, conf))
            return list_faces
    
class EmbeddingModel:
    def __init__(self, name: str):
        self.__valid_name__ = ["Custom", "ArcFace", "Facenet512"] # VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, SFace and GhostFaceNet
        self.get_model(name)
        
    def get_model(self, model_name):
        if model_name not in self.__valid_name__:
            print(f"Invalid model name: {model_name}. Choose one of {self.__valid_name__}")
            return 
        if model_name == "Custom":
            self.model = CustomEmbedding()
            self.model.__name__ = model_name
            self.model.__threshold__ = 0.5
        elif model_name == "ArcFace":
            self.model = DeepFace
            self.model.__name__ = model_name
            self.model.__threshold__ = 0.68
        elif model_name == "Facenet512":
            self.model = DeepFace
            self.model.__name__ = model_name
            self.model.__threshold__ = 0.3
            
    def get_embedding(self, image) -> np.array:
        """
        Args:
            image (np.array): image to get embedding from
            
        Returns:
            np.array: embedding vector
        """
        if self.model.__name__ == "Custom":
            return CustomEmbedding.extract_feature(image)
        elif self.model.__name__ in ["ArcFace", "Facenet512"]:
            return DeepFace.represent(
                model_name=self.model.__name__,
                detector_backend='skip',
                enforce_detection=False
                )
        else:
            print("Invalid model")
            return np.array([])
        
    