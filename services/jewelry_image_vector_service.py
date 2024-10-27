from lib.supabase_client import SupabaseClient


class JewelryImageVectorService:
    def __init__(self):
        self.supabase_client = SupabaseClient().client
        self.table_name = 'jewelry_image_vectors'

    def save(self, data):
        try:
            if not data:
                raise ValueError("Data is missing")

            response = self.supabase_client.table(self.table_name).insert(data).execute()
            print(str(response))
            if response.data:
                return response.data
            else:
                raise ValueError(f"Error inserting data into {self.table_name} table: {response.error}")
        except Exception as e:
            raise ValueError(f"Error inserting data into {self.table_name} table: {str(e)}")

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
