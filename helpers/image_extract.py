from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from lib.supabase_client import SupabaseClient


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs,
                          outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, img):
    """
    Extracts a feature vector from an image.

    Parameters:
    - model: The pre-trained model used to extract features.
    - img: A PIL Image object.

    Returns:
    - vector: The normalized feature vector extracted from the image.
    """
    print("Processing extract vector")
    img_tensor = image_preprocess(img)
    # Extract features
    vector = model.predict(img_tensor)[0]
    # Normalize the vector (L2 normalization)
    vector = vector / np.linalg.norm(vector)
    return vector


class ImageExtractService:
    def __init__(self):
        # Initialize Supabase client
        self.supabase_client = SupabaseClient()
        # Load the extraction model
        self.model = get_extract_model()

    def download_and_process_image(self, image_url: str) -> np.ndarray:
        """Download image from Supabase storage and create embedding."""
        try:
            # Fetch image from URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raises an error for bad status codes
            # Convert bytes to image
            img = Image.open(BytesIO(response.content)).convert('RGB')
            # Preprocess the image and extract vector
            image_vector = extract_vector(self.model, img)
            return image_vector
        except requests.RequestException as req_err:
            raise Exception(f"Error downloading image: {req_err}")
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def image_to_vector(self, image_url: str) -> np.ndarray:
        """Convert image URL to vector using model."""
        image_vector = self.download_and_process_image(image_url)
        return image_vector
