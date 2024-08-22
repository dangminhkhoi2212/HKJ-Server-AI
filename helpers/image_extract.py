import numpy as np
from PIL import Image
from keras import Model
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs,
                          outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# def extract_vector(model, image_path):
#     print("Xu ly : ", image_path)
#     img = Image.open(image_path)
#     img_tensor = image_preprocess(img)
#
#     # Trich dac trung
#     vector = model.predict(img_tensor)[0]
#     # Chuan hoa vector = chia chia L2 norm (tu google search)
#     vector = vector / np.linalg.norm(vector)
#     return vector


def extract_vector(model, image_file):
    """
    Extracts a feature vector from an image.

    Parameters:
    - model: The pre-trained model used to extract features.
    - image_file: A file-like object containing the image.

    Returns:
    - vector: The normalized feature vector extracted from the image.
    """
    print("Processing extract vector")
    img = Image.open(image_file)
    img_tensor = image_preprocess(img)

    # Extract features
    vector = model.predict(img_tensor)[0]

    # Normalize the vector (L2 normalization)
    vector = vector / np.linalg.norm(vector)

    return vector
