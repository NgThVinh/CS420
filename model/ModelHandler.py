import numpy as np
import cv2
from model import CustomEmbedding
from deepface import DeepFace
from deepface.modules.modeling import build_model 
from deepface.modules.detection import detect_faces
from deepface.modules.verification import find_distance, find_threshold


# Base Source: DeepFace, this function is modified to fit the project
def resize_image(img: np.ndarray, target_size) -> np.ndarray:
    """
    Resize an image to expected size of a ml model with adding black pixels.
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
    Returns:
        img (np.ndarray): resized input image
    """
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img = np.array(img).astype(np.uint8)

    return img

# Base Source: DeepFace, this function is modified to fit the project
def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image.
    """
    
    img = img.astype(np.float32)
    
    if normalization == "base":
        img = (img / 255.0).astype(np.float32)
        return img
    
    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        # mean subtraction based on VGGFace2 training data
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        # Reference study: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    else:
        img = (img / 255.0).astype(np.float32)

    return img

class DetectionModel:
    def __init__(self, name: str):
        self.__valid_name__ = ['opencv', 'retinaface', 'mtcnn', 'ssd', 
                               'dlib', 'mediapipe', 'yolov8', 'centerface']
        self.get_model(name)
        
    def get_model(self, model_name) -> None:
        if model_name not in self.__valid_name__:
            print(f"Invalid model name: {model_name}")
            return
        self.__name__ = model_name
    
    def __call__(self, image, align=True) -> list:
        if not self.__name__:
            print("No model is loaded.")
            return []
        face_objs = detect_faces(img=image, 
                                 detector_backend=self.__name__, 
                                 align=align)
        if len(face_objs) == 0:
            return []
        return list((face.img, face.confidence) for face in face_objs)
    
class EmbeddingModel:
    def __init__(self, name: str):
        self.__valid_name__ = ["Custom", "VGG-Face", "Facenet", "Facenet512",
                               "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", 
                               "SFace", "GhostFaceNet"]
        self.model = None
        self.input_shape = None
        self.get_model(name)
        
    def get_model(self, model_name):
        if model_name not in self.__valid_name__:
            print(f"Invalid model name: {model_name}. Choose one of {self.__valid_name__}")
            return 
        if model_name == "Custom":
            self.model = CustomEmbedding()
            self.model.__name__ = model_name
        else:
            model_client = build_model(
                task="facial_recognition", 
                model_name=model_name
            )
            self.model = model_client.model
            self.input_shape = model_client.input_shape
            self.model.__name__ = model_name

    def preprocess_input(self, image):
        if not self.model:
            return image
        if self.model.__name__ == "Custom":
            return image
        image = resize_image(image, target_size=self.input_shape)
        image = normalize_input(image, normalization=self.model.__name__)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
            
    def get_embedding(self, image, batch=False, preprocess_input=True) -> list:
        if not self.model:
            return None
        if self.model.__name__ == "Custom":
            return CustomEmbedding.extract_feature(image)
        else:
            input_img = image
            if not batch:
                input_img = np.expand_dims(input_img, axis=0) # make it 4-dimensional how ML models expect
            if preprocess_input:
                input_img = np.array([self.preprocess_input(img) for img in input_img])
            if batch:
                return self.model(input_img, training=False).numpy().tolist()
            return self.model(input_img, training=False).numpy()[0].tolist()

        
    