import time
from math import sqrt
from typing import Dict, List, Tuple

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
        self.vector_service = JewelryImageVectorService()
        self.distance_threshold: float = 1.2  # Set default if MAX_DISTANCE is not set
        self.K = 10
        self.table_name = TABLE_VECTORS
        self.image_ids = []

    def build_faiss_index(self):
        """Build a FAISS index with vectors from Supabase."""

        index_file = load_file(name.FILE_FAISS_INDEX)
        if index_file is not None:
            return index_file

        print('Load vector from Supabase...')
        ids, vectors = self.vector_service.load_vectors_from_supabase()
        if len(vectors) == 0:
            raise ValueError("No vectors found in Supabase")

        dimension = vectors.shape[1]
        nlist = round(sqrt(len(vectors)))  # Số cụm (centroids), bạn có thể tùy chỉnh
        quantizer = faiss.IndexFlatL2(dimension)  # Chỉ số cơ sở dùng cho phân cụm
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # Huấn luyện index với dữ liệu vector
        index.train(vectors)
        print('Index trained successfully.')

        # Gắn id vào index
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
            # Thiết lập nprobe để tìm kiếm trong nhiều cụm
            index.nprobe = 20
            distances, indices = index.search(search_vector, self.K)
            print(f'distances: {distances}')
            # Create a list of nearest images with distances as percentages
            nearest_images = [
                {
                    "path": idx,
                    "distance": dist
                }
                for idx, dist in zip(indices[0], distances[0])
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

    def get_data_from_supabase(self):
        def convert_data_to_json(id, url, jewelryId):
            return {
                'id': id,
                'url': url,
                'jewelryId': jewelryId
            }

        supabase = self.supabaseClient
        jewelry_models = supabase.client.table('hkj_jewelry_model').select('*').eq('is_deleted', False).eq(
            'is_cover_search',
            False).execute()
        jewelry_images = supabase.client.table('hkj_jewelry_image').select('*').eq('is_deleted', False).eq(
            'is_search_image', False).execute()
        jewelry_convert = [convert_data_to_json(item['id'], item['cover_image'], item['id']) for item in
                           jewelry_models.data]
        jewelry_image_convert = [
            convert_data_to_json(item['id'], item['url'], item['jewelry_model_id']) for item in jewelry_images.data
        ]
        return jewelry_convert + jewelry_image_convert

    def evaluate_faiss_ivf(self,
                           vectors: np.ndarray,
                           queries: np.ndarray,
                           true_labels: np.ndarray,
                           nlist_values: List[int],
                           nprobe_values: List[int],
                           top_k: int = 10
                           ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Khảo sát hiệu suất của FAISS IndexIVFFlat khi thay đổi các tham số `nlist` và `nprobe`.

        Args:
            vectors (np.ndarray): Bộ dữ liệu vector (được dùng để xây dựng FAISS index).
            queries (np.ndarray): Các vector truy vấn.
            true_labels (np.ndarray): Nhãn thực sự của kết quả tìm kiếm (cho đánh giá độ chính xác).
            nlist_values (List[int]): Danh sách giá trị `nlist` cần thử nghiệm.
            nprobe_values (List[int]): Danh sách giá trị `nprobe` cần thử nghiệm.
            top_k (int): Số lượng kết quả trả về hàng đầu để đánh giá.

        Returns:
            Dict[Tuple[int, int], Dict[str, float]]: Kết quả khảo sát với các chỉ số hiệu suất cho từng cặp tham số.
        """
        results = {}

        for nlist in nlist_values:
            quantizer = faiss.IndexFlatL2(vectors.shape[1])
            index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_L2)

            # Huấn luyện index với bộ vector
            index.train(vectors)
            index.add(vectors)

            for nprobe in nprobe_values:
                index.nprobe = nprobe

                start_time = time.time()
                distances, indices = index.search(queries, top_k)
                end_time = time.time()

                # Tính độ chính xác
                accuracy = np.mean([1 if true_labels[i] in indices[i] else 0 for i in range(len(queries))])

                # Tính thời gian thực thi
                execution_time = end_time - start_time

                # Tính khoảng cách trung bình
                avg_distance = np.mean(distances)

                # Lưu kết quả
                results[(nlist, nprobe)] = {
                    "accuracy": accuracy,
                    "execution_time": execution_time,
                    "avg_distance": avg_distance
                }

                print(
                    f"nlist: {nlist}, nprobe: {nprobe} -> Accuracy: {accuracy:.4f}, Time: {execution_time:.4f}s, "
                    f"Avg Distance: {avg_distance:.4f}")

        return results


def main():
    image_search_service = ImageSearchService()
    # Dữ liệu mẫu
    url = r"C:\Users\WINDOWS\Desktop\on-gn0000y003164-nhan-vang-24k-pnj-871.jpg"
    image_search_service.build_faiss_index()
    # image = Image.open(url)
    # image.show()
    # result = image_search_service.search_image(image)
    # print(result)
    # np.random.seed(0)
    # data_vectors = np.random.random((10000, 4096)).astype('float32')
    # query_vectors = np.random.random((100, 4096)).astype('float32')
    # true_labels = np.random.randint(0, 10000, size=(100,))
    #
    # # Thực hiện khảo sát
    # nlist_options = [50, 100, 200]
    # nprobe_options = [5, 10, 20]
    # results = image_search_service.evaluate_faiss_ivf(data_vectors, query_vectors, true_labels, nlist_options,
    #                                                   nprobe_options)
    #
    # # Tìm tham số tối ưu
    # best_params = max(results.items(), key=lambda x: x[1]['accuracy'])
    # print(
    #     f"Best Params: nlist={best_params[0][0]}, nprobe={best_params[0][1]} with Accuracy="
    #     f"{best_params[1]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
