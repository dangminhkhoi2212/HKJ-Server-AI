import asyncio
from math import sqrt

import faiss

from constants import name
from helpers.file_helper import load_file, save_file
from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.jewelry_image_vector_service import JewelryImageVectorService


class BuildFaissService:
    def __init__(self):
        self.supabase = SupabaseClient().client
        self.extract_service = ImageExtractService()
        self.jewelry_image_vector_service = JewelryImageVectorService()

    def _handle_build_faiss_index(self, ids, vectors):
        print('Load vector from Supabase...')

        if len(vectors) == 0:
            raise ValueError("No vectors found in Supabase")

        dimension = vectors.shape[1]
        nlist = round(sqrt(len(vectors)))  # Số cụm (centroids), có thể tùy chỉnh
        quantizer = faiss.IndexFlatL2(dimension)  # Chỉ số cơ sở dùng cho phân cụm
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # Huấn luyện index với dữ liệu vector
        index.train(vectors)
        print('Index trained successfully.')

        # Gắn id vào index
        # index = faiss.IndexIDMap(index)
        index.add_with_ids(vectors, ids)

        save_file(index, name.FILE_FAISS_INDEX)

        print('Build FAISS index successfully')
        return index

    def build_faiss_index(self):
        """Build a FAISS index with vectors from Supabase."""
        index_file = load_file(name.FILE_FAISS_INDEX)
        if index_file is not None:
            return index_file
        ids, vectors = self.jewelry_image_vector_service.load_vectors_from_supabase()
        print(f'loaded ids: {ids}')
        if len(vectors) == 0:
            raise ValueError("No vectors found in Supabase")
        return self._handle_build_faiss_index(ids, vectors)

    async def get_vectors_and_build_index(self):
        data = self.jewelry_image_vector_service.get_data_from_supabase()
        print(f"Get {len(data)} vectors from Supabase")
        if len(data) > 0:
            ## Trích đặc trưng ảnh và lưu lên db
            self.extract_service.process_image(data)
            ## Lấy ảnh từ db về


async def main():
    buidl_faiss_service = BuildFaissService()
    await buidl_faiss_service.get_vectors_and_build_index()


if __name__ == '__main__':
    asyncio.run(main())
