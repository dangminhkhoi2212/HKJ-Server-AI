import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import sqrt
from typing import Dict, List, Tuple

import faiss
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from constants.name import TABLE_VECTORS
from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.jewelry_image_vector_service import JewelryImageVectorService

load_dotenv()
from services.build_faiss_service import BuildFaissService

build_faiss_service = BuildFaissService()


class ImageSearchService:
    def __init__(self):
        self.supabaseClient = SupabaseClient()
        self.image_extract_service = ImageExtractService()
        self.vector_service = JewelryImageVectorService()
        self.distance_threshold: float = 1.2  # Set default if MAX_DISTANCE is not set
        self.K = 10
        self.table_name = TABLE_VECTORS
        self.image_ids = []
        self.index = build_faiss_service.build_faiss_index()

    # def build_faiss_index(self):
    #     """Build a FAISS index with vectors from Supabase."""
    #     if self.index is not None:
    #         print('Loaded index from service')
    #         return self.index
    #     index_file = load_file(name.FILE_FAISS_INDEX)
    #     if index_file is not None:
    #         self.index = index_file
    #         return index_file
    #
    #     print('Load vector from Supabase...')
    #     ids, vectors = self.vector_service.load_vectors_from_supabase()
    #     print(f'loaded ids: {ids}')
    #     if len(vectors) == 0:
    #         raise ValueError("No vectors found in Supabase")
    #
    #     dimension = vectors.shape[1]
    #     nlist = round(sqrt(len(vectors)))  # Số cụm (centroids), bạn có thể tùy chỉnh
    #     quantizer = faiss.IndexFlatL2(dimension)  # Chỉ số cơ sở dùng cho phân cụm
    #     index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    #
    #     # Huấn luyện index với dữ liệu vector
    #     index.train(vectors)
    #     print(f'Index trained successfully: {index.is_trained}')
    #
    #     # Gắn id vào index
    #     index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
    #
    #     save_file(index, name.FILE_FAISS_INDEX)
    #     self.index = index
    #     print('Build FAISS index successfully')
    #     return index

    def search_with_faiss(self, image, k=10):

        search_vector = self.image_extract_service.extract_vector(image)
        search_vector = np.atleast_2d(search_vector)
        index = self.index
        print(f'ntotal = {round(sqrt(index.ntotal))}')
        # Thiết lập nprobe để tìm kiếm trong nhiều cụm
        # index.nprobe = round(0.3 * sqrt(index.ntotal))
        # index.nprobe = round(0.5 * sqrt(index.ntotal))
        # print('nprobe = {}'.format(index.nprobe))
        # index.nprobe = round(0.7 * sqrt(index.ntotal))
        index.nprobe = round(sqrt(index.ntotal))

        print(f"nprobe = {index.nprobe}")
        distances, indices = index.search(search_vector, k)
        # Create a list of nearest images with distances as percentages
        nearest_images = [
            {
                "id": idx,
                "distance": dist
            }
            for idx, dist in zip(indices[0], distances[0])
        ]

        return nearest_images

    def sorted_field(self, x):
        return x['distance']

    def search_image(self, file, k=10) -> List[Dict]:
        """Search image in Supabase storage and return results."""
        try:
            # Convert to image without background
            file_bytes = file.read()
            image_no_bg = self.image_extract_service.remove_bg_image(file_bytes)

            # Open the original image
            image = Image.open(BytesIO(file_bytes)).convert('RGB')

            if image is None and image_no_bg is None:
                raise ValueError('Image not found')

            # Use ThreadPoolExecutor to perform searches in parallel
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                future_no_bg = executor.submit(self.search_with_faiss, image_no_bg, k)
                future_with_bg = executor.submit(self.search_with_faiss, image, k)

                # Collect results as they complete
                search_no_bg = future_no_bg.result()
                search_with_bg = future_with_bg.result()
            end_time = time.time()  # Kết thúc đo thời gian
            elapsed_time = end_time - start_time  # Tính toán thời gian đã trôi qua
            print(f"Thời gian thực hiện tìm kiếm: {elapsed_time:.4f} giây")  # In ra thời gian thực hiện

            # Combine and sort results
            result = search_with_bg + search_no_bg
            result = sorted(result, key=self.sorted_field)
            print(f'result: {result}')

            # Unique ids
            # Unique ids while maintaining order
            seen = set()
            unique_ids = []
            for item in result:
                if item['id'] not in seen:
                    seen.add(item['id'])
                    unique_ids.append(item['id'])

            print(f'Unique ids: {unique_ids}')

            # Fetch jewelry models from Supabase
            jewelry_models = self.supabaseClient.find_jewelry(unique_ids)
            ids_search = [jewelry['category_id'] for jewelry in jewelry_models]
            print(f'jewelry_models: {ids_search}')
            return jewelry_models

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
