import io
import math

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from constants import name
from helpers import image_extract
from helpers.file_helper import load_file
from lib.supabase_client import SupabaseClient
from routes.search_image import supabase_client


def load_faiss_index(supabase_client, file_name: str):
    try:
        # Download the FAISS index file content from Supabase Storage
        file_name, file_content = supabase_client.get_one_file_from_bucket(file_name, 'models')

        if file_content:
            # Load the FAISS index from the file content
            index_stream = io.BytesIO(file_content)
            index = faiss.read_index(index_stream)
            print("Successfully loaded FAISS index")
            return index
        else:
            raise ValueError("File content is empty or could not be retrieved.")
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        return None


def load_vectors_and_paths_and_faiss():
    try:
        vectors = load_file(name.FILE_VECTORS)
        paths = load_file(name.FILE_PATHS)
        index = faiss.read_index(name.FILE_FAISS_INDEX)
        # Ensure vectors are in the correct shape
        # if len(vectors.shape) == 1:
        #     vectors = vectors.reshape(1, -1)
        # if len(paths.shape) == 1:
        #     paths = paths.reshape(1, -1)
        return vectors, paths, index
    except Exception as e:
        print(f"Error loading vectors or paths: {str(e)}")
        return None, None, None


def search_image(index, search_vector, k=12):
    if index:
        # Perform the search with FAISS
        distances, indices = index.search(search_vector, k)
        return distances, indices
    return None, None


def display_nearest_images(paths, indices, distances, k=12):
    # Ensure that paths, indices, and distances are not empty
    if len(paths) > 0 and len(indices) > 0 and len(distances) > 0:
        # Limit k to the number of available images
        k = min(k, len(indices[0]), len(paths))

        # Determine the number of rows and columns for the grid
        columns = math.ceil(math.sqrt(k))
        rows = math.ceil(k / columns)

        # Display the nearest images
        fig, axes = plt.subplots(rows, columns, figsize=(15, 5))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i in range(k):
            # Ensure the index is within bounds
            if indices[0][i] < len(paths):
                img_path = paths[indices[0][i]]
                file_name, image = supabase_client.get_one_file_from_bucket(img_path, name.BUCKET_IMAGES)
                image_stream = io.BytesIO(image)
                img = Image.open(image_stream)
                dist = distances[0][i]
                axes[i].set_title(f"Distance: {dist:.4f}", fontsize=8)
                axes[i].imshow(img)
                axes[i].axis('off')
            else:
                axes[i].axis('off')  # Turn off unused axes

        # Turn off any unused axes
        for j in range(k, len(axes)):
            axes[j].axis('off')

        fig.tight_layout()
        plt.show()
    else:
        print("No images or indices to display.")


def main():
    supabase_client = SupabaseClient()
    model_name = "faiss_index.index"
    search_image_path = "testimage/test1.jpg"

    # Initialize model
    model = image_extract.get_extract_model()

    # Extract features for the search image
    search_vector = image_extract.extract_vector(model, search_image_path)
    search_vector = np.expand_dims(search_vector, axis=0)  # Ensure it's a 2D array

    # Load vectors and paths and index
    vectors, paths, index = load_vectors_and_paths_and_faiss()
    print(len(paths), len(vectors))
    # Search the image
    distances, indices = search_image(index, search_vector, k=5)
    print('distances: ', distances)
    print('indices: ', indices)
    # Display the nearest images
    display_nearest_images(paths, indices, distances, k=5)


if __name__ == "__main__":
    main()
