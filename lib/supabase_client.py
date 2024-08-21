# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values
# loading variables from .env file
from supabase import create_client, Client

load_dotenv()
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')


class SupabaseClient:
    def __init__(self):
        self.client: Client = create_client(url, key)

    def insert_table(self, data, table_name: str):
        response = self.client.table(table_name).insert(data).execute()
        return response

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
            if res.error:
                print(
                    f"Error deleting bucket '{bucket_name}': {res.error.message}")
            else:
                print(f"Successfully deleted bucket '{bucket_name}'.")
        except Exception as e:
            print(
                f"An exception occurred while deleting bucket '{bucket_name}':{str(e)}")
