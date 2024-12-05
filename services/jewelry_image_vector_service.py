import numpy as np

from constants.name import TABLE_VECTORS
from lib.supabase_client import SupabaseClient


class JewelryImageVectorService:
    def __init__(self):
        self.supabase_client = SupabaseClient().client
        self.table_name = TABLE_VECTORS

    def load_vectors_from_supabase(self):
        """Retrieve all `id` and `vector` from Supabase without limit."""
        all_records = []
        offset = 0
        limit = 1000  # Supabase default limit
        while True:
            print(f"Fetching records from {offset} to {offset + limit - 1}...")
            response = (
                self.supabase_client
                .table(self.table_name)
                .select("id", "jewelry_id", "vector")
                .eq("has_vector", True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            if response.data is None:
                raise Exception(f"Supabase query failed: {response.error}")

            all_records.extend(response.data)
            # Nếu số lượng bản ghi nhỏ hơn limit, nghĩa là đã lấy hết
            if len(response.data) < limit:
                break

            # Tăng offset cho lần truy vấn tiếp theo
            offset += limit

        ids = [record["id"] for record in all_records]
        vectors = [record["vector"] for record in all_records]

        return ids, np.array(vectors, dtype="float32")

    def save(self, data):
        try:
            if not data:
                raise ValueError("Data is missing")

            response = self.supabase_client.table(self.table_name).insert(data).execute()

            if response.data:
                return response.data
            else:
                raise ValueError(f"Error inserting data into {self.table_name} table: {response.error}")
        except Exception as e:
            raise ValueError(f"Error inserting data into {self.table_name} table: {str(e)}")

    def update_is_searched(self, jewelry_id):
        try:
            ## update search in jewelry_model
            self.supabase_client.table('hkj_jewelry_model').update({'is_cover_search': True}).eq(
                'id',
                jewelry_id).execute()
            ## update search in jewelry_images
            self.supabase_client.table('hkj_jewelry_image').update({'is_search_image': True}).eq(
                'jewelry_model_id',
                jewelry_id).execute()

        except Exception as e:
            raise ValueError(f"Error updating is_searched: {str(e)}")

    def is_extracted(self, image_id):
        try:
            response = self.supabase_client.table(self.table_name).select('has_vector').eq('image_id',
                                                                                           image_id).execute()
            data = response.data

            print(f'jewelry_image_vectors: {data}')
            if len(data) > 0:
                return data[0].get('has_vector')
            else:
                return False
        except Exception as e:
            raise ValueError(f"Error checking if image is extracted: {str(e)}")

    def remove_images(self, images):
        try:
            if len(images) == 0:
                raise ValueError("No images to remove")
            response = self.supabase_client.table(self.table_name).delete().in_('image_id', images).execute()
            return response.data
        except Exception as e:
            raise ValueError(f"Error removing images: {str(e)}")

    # def load_vectors_from_supabase(self):
    #     """Retrieve `id` and `vector` from Supabase."""
    #     response = self.supabase_client.table(self.table_name).select("jewelry_id, vector").eq("has_vector",
    #                                                                                            True).execute()
    #     if response.data is None:
    #         raise Exception(f"Supabase query failed: {response.error}")
    #
    #     ids = []
    #     vectors = []
    #     for record in response.data:
    #         ids.append(record["jewelry_id"])
    #         vectors.append(record["vector"])  # Convert from JSON string to array
    #
    #     return ids, np.array(vectors, dtype="float32")

    def get_data_from_supabase(self):
        def convert_data_to_json(id, url, jewelryId):
            return {
                'id': id,
                'url': url,
                'jewelryId': jewelryId
            }

        supabase = self.supabase_client
        jewelry_models = supabase.table('hkj_jewelry_model').select('*').eq('is_deleted', False).eq(
            'is_cover_search',
            False).execute()
        jewelry_images = supabase.table('hkj_jewelry_image').select('*').eq('is_deleted', False).eq(
            'is_search_image', False).execute()
        jewelry_convert = [convert_data_to_json(item['id'], item['cover_image'], item['id']) for item in
                           jewelry_models.data]
        jewelry_image_convert = [
            convert_data_to_json(item['id'], item['url'], item['jewelry_model_id']) for item in jewelry_images.data
        ]
        print(f'jewelry_convert: {len(jewelry_convert)}')
        print(f'jewelry_image_convert: {len(jewelry_image_convert)}')
        return jewelry_convert + jewelry_image_convert


def main():
    service = JewelryImageVectorService()
    ids, vectors = service.build_faiss_index()
    print(len(vectors))


if __name__ == '__main__':
    main()
