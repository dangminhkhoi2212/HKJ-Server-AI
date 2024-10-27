from io import BytesIO

import faiss
import numpy as np
import requests
from PIL import Image

from constants import name
from helpers.file_helper import save_file
from helpers.image_extract import get_extract_model, extract_vector
from lib.supabase_client import SupabaseClient
from services.jewelry_image_vector_service import JewelryImageVectorService


class ImageExtractService:
    def __init__(self):
        # Initialize Supabase client
        self.supabase_client = SupabaseClient()
        # Load the extraction model
        self.model = get_extract_model()
        self.jewelry_image_vector_service = JewelryImageVectorService()

    def download_and_process_image(self, image_url: str) -> np.ndarray:
        """Download image from Supabase storage and create embedding."""
        try:
            # Fetch image from URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raises an error for bad status codes

            # Convert bytes to image
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Preprocess the image and extract vector
            image_vector = extract_vector(self.model, image)
            return image_vector
        except requests.RequestException as req_err:
            raise Exception(f"Error downloading image: {req_err}")
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def image_to_vector(self, image_url: str) -> np.ndarray:
        """Convert image URL to vector using model."""
        image_vector = self.download_and_process_image(image_url)
        return image_vector

    def process_image(self, images: list):
        """Convert image URL to vector using model."""
        vectors = []
        image_ids = []
        data_db = []
        for image_data in images:
            print(f'image: {image_data["url"]}')
            image_url = image_data.get('url')
            image_id = image_data.get('id')
            jewelry_id = image_data.get('jewelryId')
            if self.jewelry_image_vector_service.is_extracted(image_id) is True:
                continue

            image_vector = self.image_to_vector(image_url)
            image_ids.append(jewelry_id)
            vectors.append(image_vector)
            data_db.append({
                "image_id": image_id,
                "url": image_url,
                "vector": image_vector.tolist(),
                "jewelry_id": jewelry_id,
                "has_vector": True,
                "active": True
            })
        if len(data_db) == 0:
            return 'No new images found'
        response = self.jewelry_image_vector_service.save(data_db)
        self.save_model(np.array(vectors), image_ids)
        return response

    def save_model(self, vectors, paths) -> None:
        # Create and save a FAISS index
        try:
            # Save vectors to files
            save_file(vectors, name.FILE_VECTORS)

            # Save path to files
            save_file(paths, name.FILE_PATHS)

            # Save the FAISS index to a file
            vector_dim = vectors.shape[1]  # Vector dimensionality
            index = faiss.IndexFlatL2(vector_dim)  # Create an L2 distance FAISS index
            index.add(vectors)  # Add vectors to the index
            faiss.write_index(index, name.FILE_FAISS_INDEX)
            print("Successfully saved FAISS index to file")

        except Exception as e:
            print(f"Error handling FAISS index: {str(e)}")
