# importing os module for environment variables
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
from supabase import create_client, Client

from constants import name


class SupabaseClient:
    def __init__(self):
        load_dotenv()
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        self.client: Client = create_client(url, key)

    def olf_find_jewelry(self, ids: list):
        try:

            # Query the Supabase table for jewelry models related to the given ids

            response = self.client.table(name.TABLE_VECTORS).select(
                '''id, hkj_jewelry_model(id, name, cover_image, price, category_id)'''
            ).in_('id', ids).execute()

            jewelry_models = response.data
            # print(f"response: {jewelry_models}")
            if not jewelry_models:
                print("No jewelry models found for the provided IDs.")
                return []

            sorted_models = []
            for id in ids:
                for jewelry in jewelry_models:
                    if jewelry['id'] == id:
                        sorted_models.append(jewelry)
                        break
            result = [item['hkj_jewelry_model'] for item in sorted_models]
            result_ids = [item['id'] for item in sorted_models]
            print(f'ids: {ids}')
            print(f'result_ids: {result_ids}')
            print(f"sorted_models: {sorted_models}")

            return sorted_models

        except Exception as e:
            print(f"Error in find_jewelry: {str(e)}")
            return []

    def find_jewelry(self, ids: list):
        try:
            ids = [int(id) for id in ids]

            response = self.client.rpc('get_unique_jewelry', {'ids': ids}).execute()
            jewelry_models = response.data

            print(
                f"response: "
                f"{[{'vector_id': item['vector_id'], 'id': item['id'], 'name': item['name']} for item in jewelry_models]}")
            if not jewelry_models:
                print("No jewelry models found for the provided IDs.")
                return []
            return jewelry_models

        except Exception as e:
            print(f"Error in find_jewelry: {str(e)}")
            return []

    def create_path_bucket(self, file_name: str, bucket_name: str):
        return f"{bucket_name}/{file_name}"

    def insert_table(self, data, table_name: str):
        try:
            if not data:
                raise ValueError("Data is missing")

            response = self.client.table(table_name).insert(data).execute()
            return response
        except Exception as e:
            raise ValueError(f"Error inserting data into '{table_name}' table: {str(e)}")

    def empty_table(self, table_name: str):
        response = self.client.table(table_name).delete().neq('id',
                                                              -1).execute()
        return response

    def insert_bucket(self, file, bucket_name: str):
        try:
            with open(file, 'rb') as f:
                bucket = self.client.storage.from_(bucket_name)
                response = bucket.upload(file=f)
                return response
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{file}' not found.") from e
        except Exception as e:
            raise Exception(f"Error uploading file '{file}': {str(e)}") from e

    def empty_bucket(self, bucket_name: str):
        try:
            # Attempt to delete the bucket
            res = self.client.storage.empty_bucket(bucket_name)
            print(res)
            # if res.error:
            #     print(
            #         f"Error deleting bucket '{bucket_name}': {res.error.message}")
            # else:
            #     print(f"Successfully deleted bucket '{bucket_name}'.")
        except Exception as e:
            print(
                f"An exception occurred while deleting bucket '{bucket_name}':{str(e)}")

    def get_one_file_from_bucket(self, file_name: str, bucket_name: str):
        try:
            response = self.client.storage.from_(bucket_name).download(file_name)
            return response
        except Exception as e:
            print(f"Error retrieving '{file_name}': {str(e)}")
            return None

    def get_one_file_from_bucket_by_url(self, file_name: str, bucket_name: str):
        try:
            response = self.client.storage.from_(bucket_name).get_public_url(file_name)

            return response
        except Exception as e:
            print(f"Error retrieving '{file_name}': {str(e)}")
            return None

    def get_list_files_from_bucket(self, bucket_name: str):
        try:
            all_files = []
            last_file_name = None

            while True:
                options = {
                    'limit': 100,
                    'offset': len(all_files)  # Offset to paginate
                }

                response = self.client.storage.from_(bucket_name).list(options=options)

                if response:
                    files = [file['name'] for file in response]
                    all_files.extend(files)

                    if len(files) < 100:
                        # If less than 100 files were returned, we've reached the end
                        break
                else:
                    print(f"No files found in bucket '{bucket_name}'.")
                    break

            return all_files

        except Exception as e:
            print(f"Error listing files in bucket '{bucket_name}': {str(e)}")
            return []

    def get_multiple_files_from_bucket(self, bucket_name: str, file_names: list = None, max_workers=5):
        # If no file names are provided, retrieve all files from the bucket
        if file_names is None:
            file_names = self.get_list_files_from_bucket(bucket_name)

        files_content = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.get_one_file_from_bucket, file_name, bucket_name): file_name for
                              file_name in
                              file_names}

            for future in as_completed(future_to_file):
                file_name, content = future.result()
                files_content[file_name] = content

        return files_content

    def upload_bucket(self, file, bucket_name: str):
        try:
            # Lấy thời gian hiện tại và định dạng nó để tạo tên file mới
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = file.filename.split('.')[-1]  # Lấy phần mở rộng của file (ví dụ: png, jpg)
            new_file_name = f"{current_time}.{extension}"  # Tạo tên file mới với thời gian

            # Đọc nội dung file
            file_content = file.read()

            # Tải file lên bucket
            response = self.client.storage.from_(bucket_name).upload(
                path=new_file_name,
                file=file_content,
                file_options={"content-type": "image/png"}
            )

            print(f"Successfully uploaded file '{new_file_name}' to bucket '{bucket_name}'")
            return response.url

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found.", str(e))
        except Exception as e:
            raise Exception(f"Error uploading file : {str(e)}")

    def delete_files_from_bucket(self, bucket_name: str, file_names: list = None):
        response = self.client.storage.from_(bucket_name).remove(file_names)
        return response

    def delete_table(self, column: str, value, table_name: str):
        return self.client.table(table_name).delete().eq(column, value).execute()
