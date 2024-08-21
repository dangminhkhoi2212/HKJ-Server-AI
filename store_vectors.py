import os
import pickle
import image_extract
import numpy as np
import faiss
from supabase import create_client, Client
# Initialize Supabase client
url = "https://ckduvpcvhlbgrujnrftb.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNrZHV2cGN2aGxiZ3J1am5yZnRiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMzQ2MDI2MywiZXhwIjoyMDM5MDM2MjYzfQ.Kwa1bfhFeGsqpDzHGzYLwYdYPRebAV6R5JwOXTkPWUs"
supabase: Client = create_client(url, key)

def empty_bucket(bucket_name):
    try:
        files = supabase.storage().from_(bucket_name).list().data
        if files:
            for file in files:
                delete_response = supabase.storage().from_(bucket_name).remove([file['name']])
                if delete_response.error:
                    print(f"Error deleting file {file['name']} from {bucket_name}: {delete_response.error.message}")
                else:
                    print(f"Successfully deleted file {file['name']} from {bucket_name}")
        else:
            print(f"No files found in bucket '{bucket_name}'")
    except Exception as e:
        print(f"Error emptying bucket '{bucket_name}': {str(e)}")

# Delete all data in the 'images' table

delete_response = supabase.table('images').delete().neq('id', -1).execute()
print(delete_response)
# if delete_response.status_code != 200:
#     print(f"Error deleting data from 'images' table: {delete_response.status_text}")
# else:
#     print(f"Successfully deleted data from 'images' table")
# Empty the 'models' and 'images' buckets
empty_bucket('models')
empty_bucket('images')

data_folder = "dataset"

# Initialize the image feature extraction model
try:
    model = image_extract.get_extract_model()
    print("Model initialized successfully")
except Exception as e:
    print(f"Error initializing model: {str(e)}")

vectors = []
paths = []

# Process images in the dataset folder
for idx, image_name in enumerate(os.listdir(data_folder)):
    if idx >= 2:  # Limiting to 2 images, remove or change as needed
        break

    image_path = os.path.join(data_folder, image_name)
    try:
        image_vector = image_extract.extract_vector(model, image_path)
        vectors.append(image_vector)
        paths.append(image_path)
        print(f"Successfully processed {image_path}")

        # Store path and vector in Supabase
        data = {
            "path": image_path,
            "vector": image_vector.tolist()  # Convert numpy array to list for JSON storage
        }
        insert_response = supabase.table("images").insert(data).execute()
        if len(insert_response.data) == 0:
            print(f"Error inserting data into 'images' table: {insert_response.data}")
        else:
            print(f"Successfully inserted {image_path} into 'images' table")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Convert list of vectors to a numpy array for FAISS
vectors = np.array(vectors)

# Save the vectors and paths to files for local use (optional)
try:
    pickle.dump(vectors, open("vectors.pkl", "wb"))
    pickle.dump(paths, open("paths.pkl", "wb"))
    print("Successfully saved vectors and paths to local files")
except Exception as e:
    print(f"Error saving vectors or paths to files: {str(e)}")

# Create and save a FAISS index
try:
    vector_dim = vectors.shape[1]  # Vector dimensionality
    index = faiss.IndexFlatL2(vector_dim)  # Create an L2 distance FAISS index

    index.add(vectors)  # Add vectors to the index
    print("Successfully created FAISS index")

    # Save the FAISS index to a file
    faiss.write_index(index, "faiss_index.index")
    print("Successfully saved FAISS index to file")

    # Upload the FAISS index to Supabase Storage
    with open("faiss_index.index", "rb") as f:
        upload_response = supabase.storage.from_("models").upload("faiss_index.index", f)
        if upload_response.error:
            print(f"Error uploading FAISS index to Supabase Storage: {upload_response.error.message}")
        else:
            print("Successfully uploaded FAISS index to Supabase Storage")

except Exception as e:
    print(f"Error handling FAISS index: {str(e)}")
