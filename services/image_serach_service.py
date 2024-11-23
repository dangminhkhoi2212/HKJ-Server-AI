import os
from typing import List, Dict

import faiss
import numpy as np
from dotenv import load_dotenv

from constants import name
from constants.name import TABLE_VECTORS
from helpers.file_helper import load_file, save_file
from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.jewelry_image_vector_service import JewelryImageVectorService

load_dotenv()


class ImageSearchService:
    def __init__(self):
        self.supabaseClient = SupabaseClient()
        self.image_extract_service = ImageExtractService()
        self.jewelryImageVectorService = JewelryImageVectorService()
        self.distance_threshold: float = float(os.getenv('MAX_DISTANCE', 1.0))  # Set default if MAX_DISTANCE is not set
        self.K = 10
        self.table_name = TABLE_VECTORS
        self.image_ids = []
        self.index = self.build_faiss_index()

    def load_vectors_from_supabase(self):
        """Retrieve `id` and `vector` from Supabase."""
        response = self.supabaseClient.client.table(self.table_name).select("jewelry_id, vector").eq("has_vector",
                                                                                                     True).execute()
        if response.data is None:
            raise Exception(f"Supabase query failed: {response.error}")

        ids = []
        vectors = []
        for record in response.data:
            ids.append(record["jewelry_id"])
            vectors.append(record["vector"])  # Convert from JSON string to array

        return ids, np.array(vectors, dtype="float32")

    def build_faiss_index(self):
        """Build a FAISS index with vectors from Supabase."""
        index_file = load_file(name.FILE_FAISS_INDEX)
        if index_file is not None:
            return index_file

        print('Load vector from Supabase...')
        ids, vectors = self.load_vectors_from_supabase()
        if len(vectors) == 0:
            raise ValueError("No vectors found in Supabase")

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(vectors, np.array(ids, dtype=np.int64))

        save_file(index, name.FILE_FAISS_INDEX)
        self.image_ids = ids
        print('Build FAISS index successfully')
        return index

    def search_image(self, file) -> List[Dict]:
        """Search image in Supabase storage and return results."""
        try:
            search_vector = self.image_extract_service.extract_vector(file)
            search_vector = np.atleast_2d(search_vector)

            index = self.build_faiss_index()
            distances, indices = index.search(search_vector, self.K)
            print(f'distances: {distances}')

            # Create a list of nearest images with distances as percentages
            nearest_images = [
                {
                    "path": idx,
                    "distance": dist * 100
                }
                for idx, dist in zip(indices[0], distances[0])
                if dist * 100 >= (self.distance_threshold * 100)
            ]

            print(f'nearest_images: {nearest_images}')

            ids = [img['path'] for img in nearest_images]
            print(f'ids: {ids}')

            # unique ids
            ids = set(ids)

            jewelry_models = self.supabaseClient.find_jewelry(ids)
            return jewelry_models.data

        except Exception as e:
            raise Exception(f"Error searching image: {str(e)}")
