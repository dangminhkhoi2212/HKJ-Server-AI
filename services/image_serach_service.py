import json
import os
from typing import List, Dict

import faiss
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from helpers import image_extract
from lib.supabase_client import SupabaseClient
from services.jewelry_image_vector_service import JewelryImageVectorService

load_dotenv()


class ImageSearchSystem:
    def __init__(self):
        load_dotenv()
        self.supabaseClient = SupabaseClient()
        # Initialize Supabase client
        self.model = image_extract.get_extract_model()
        self.jewelryImageVectorService = JewelryImageVectorService()
        self.distance_threshold: float = float(os.getenv('MAX_DISTANCE')) or 1.0
        self.K = 15
        self.table_name = 'jewelry_image_vectors'
        self.image_ids = []
        # Initialize model and transform

    def load_vectors_from_supabase(self):
        """Retrieve `id` and `vector` from Supabase."""
        response = self.supabaseClient.client.table(self.table_name).select("image_id, vector").eq("has_vector",
                                                                                                   True).execute()
        if response.data is None:
            raise Exception(f"Supabase query failed: {response.error}")

        ids = []
        vectors = []
        for record in response.data:
            ids.append(record["image_id"])
            vectors.append(json.loads(record["vector"]))  # Convert from JSON string to array

        return ids, np.array(vectors, dtype="float32")

    def build_faiss_index(self):
        """Build a FAISS index with vectors from Supabase."""
        ids, vectors = self.load_vectors_from_supabase()
        if len(vectors) == 0:
            raise ValueError("No vectors found in Supabase")
        # Initialize a FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to the index
        index.add(vectors)

        # # Optionally, save the index to a file
        # faiss.write_index(index, "path_to_save_index/faiss_index_file")

        # Map indices to ids for retrieval
        self.image_ids = ids
        print('Build FAISS index successfully')
        return index

    def search_image(self, file) -> List[Dict]:
        """Search image in Supabase storage and return results."""
        try:
            # Fetch image from URL
            # image_search = Image.open(BytesIO(file)).convert('RGB')
            # file_content = file.read()
            img = Image.open(file).convert('RGB')
            search_vector = image_extract.extract_vector(self.model, img)

            # Extract features for the search image

            # # Load vectors and paths from files
            # vectors = load_file(name.FILE_VECTORS)
            # paths = load_file(name.FILE_PATHS)

            # Ensure vectors are in the correct shape
            search_vector = np.atleast_2d(search_vector)

            # Load the FAISS index
            # index = faiss.read_index(name.FILE_FAISS_INDEX)
            # Tìm kiếm ảnh tương tự

            distances, indices = self.build_faiss_index().search(search_vector, self.K)
            nearest_images = [
                {"path": self.image_ids[idx],
                 "distance": round(float(dist), 5)}
                for idx, dist in zip(indices[0], distances[0])
            ]
            print(f'nearest_images: {nearest_images}')

            # Filter nearest images based on the distance threshold
            filtered_images = [img for img in nearest_images if img["distance"] <= self.distance_threshold]
            ids = [img['path'] for img in filtered_images]
            print(f'ids: {ids}')
            jewelry_models = self.supabaseClient.find_jewelry(ids)

            # print(f'jewelry_models: {jewelry_models.data}')
            # nearest_images.extend(jewelry_models)
            return jewelry_models.data
        except Exception as e:
            raise Exception(f"Error searching image: {str(e)}")
