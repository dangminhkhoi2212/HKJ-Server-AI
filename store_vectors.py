import io

import faiss
import numpy as np

import constants.name as name
from helpers import image_extract
from helpers.file_helper import save_file
from lib.supabase_client import SupabaseClient


def get_files_to_extract(supabase_client, model_extract):
    vectors = []
    paths = []
    data_folder = supabase_client.get_list_files_from_bucket('images')
    print('len(data_folder)', len(data_folder))
    image_count = 0
    max_images = 20  # Number of images to process
    file_name = ''
    # # Clear the 'images' table
    delete_response = supabase_client.empty_table(name.TABLE_IMAGES)
    for image_path in data_folder:
        # if image_count >= max_images:
        #     break
        image_count += 1
        print(f"{image_count}: Processing {image_path}")
        try:
            # Get the file content from Supabase
            file_name, file_content = supabase_client.get_one_file_from_bucket(image_path, 'images')
            image_stream = io.BytesIO(file_content)

            # Extract features from the image
            image_vector = image_extract.extract_vector(model_extract, image_stream)
            vectors.append(image_vector)
            paths.append(image_path)
            print(f"Successfully processed {file_name}")

            # Store path and vector in Supabase
            # data = {
            #     "name": file_name,
            #     "vector": image_vector.tolist()  # Convert numpy array to list for JSON storage
            # }
            # supabase_client.insert_table(data, 'images')

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    return np.array(vectors), np.array(paths)


def handle_save_model(vectors, paths, supabase_client):
    # Create and save a FAISS index
    try:
        # Save vectors to files
        save_file(vectors, name.FILE_VECTORS)

        # Save path to files
        save_file(paths, name.FILE_PATHS)

        # Save the FAISS index to a file
        vector_dim = vectors.shape[1]  # Vector dimensionality
        index = faiss.IndexFlatL2(vector_dim)  # Create an L2 distance FAISS index
        index.add(vectors)  # Add vectors to the index
        faiss.write_index(index, name.FILE_FAISS_INDEX)
        print("Successfully saved FAISS index to file")

    except Exception as e:
        print(f"Error handling FAISS index: {str(e)}")


def main():
    supabase_client = SupabaseClient()

    # delete_response = supabase_client.empty_table('images')

    # Load the model for feature extraction
    model_extract = image_extract.get_extract_model()

    # Extract image features and save them
    vectors, paths = get_files_to_extract(supabase_client, model_extract)

    # Handle the extracted image vectors
    if vectors.size > 0:  # Ensure there are vectors to process
        handle_save_model(vectors, paths, supabase_client)


if __name__ == "__main__":
    main()
