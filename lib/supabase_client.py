# importing os module for environment variables
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
from supabase import create_client, Client

load_dotenv()
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')


class SupabaseClient:
    def __init__(self):
        self.client: Client = create_client(url, key)

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
            file_name = file.filename
            image_existed = self.get_one_file_from_bucket(file_name, bucket_name)
            if image_existed is not None:
                file_url = self.client.storage.from_(bucket_name).get_public_url(file_name)
                return file_url
            file_content = file.read()  # Changed from file.file_name to file.filename
            response = self.client.storage.from_(bucket_name).upload(
                path=file_name,
                file=file_content, file_options=
                {"content-type": "image/png"})

            print(f"Successfully uploaded file '{file_name}' to bucket '{bucket_name}'")
            return response.url

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found.", str(e))
        except Exception as e:
            # Raise the exception with a descriptive message
            raise Exception(f"Error uploading file : {str(e)}")

    def delete_files_from_bucket(self, bucket_name: str, file_names: list = None):

        response = self.client.storage.from_(bucket_name).remove(file_names)
        return response
