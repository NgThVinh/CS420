import cv2
import numpy as np
from tensorflow.math import l2_normalize
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16 
# from tensorflow.keras.models import load_model

# model = load_model(EMBEDDING_MODEL_PATH,
#                    custom_objects={ 'l2_normalize': l2_normalize },
#                    safe_mode=False
#                 )

def build_embedding_model(input_shape, embedding_dim=512):
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(l2_normalize)(x)
    model = Model(base_model.input, x, name='embedding_model')
    return model

INPUT_SHAPE = (125, 94, 3)
EMBEDDING_WEIGHT_PATH = 'model/custom_embedding.weight.h5'

model = build_embedding_model(INPUT_SHAPE, 512)
model.load_weights(EMBEDDING_WEIGHT_PATH)

model.__name__ = "Custom"

def preprocess_image(face):
    input_img = cv2.resize(face, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    return input_img / 255
    
def extract_feature(img, verbose=False):
    img = preprocess_image(img)
    return model.predict(np.expand_dims(img, axis=0), verbose=verbose)

def predict_similarity(img1, img2, verbose=False):
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    embedding_1 = model.predict(np.expand_dims(img1, axis=0), verbose=verbose)
    embedding_2 = model.predict(np.expand_dims(img2, axis=0), verbose=verbose)
    cosine_similarity = np.dot(embedding_1, embedding_2.T) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
    return cosine_similarity[0][0]

  