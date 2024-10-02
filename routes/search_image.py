# modules/routes.py

import faiss
import numpy as np
from flask import Blueprint, request, jsonify, render_template

from constants import name
from helpers import image_extract
from helpers.file_helper import load_file
from lib.supabase_client import SupabaseClient

# Create a Blueprint for the routes
search_image = Blueprint('search_image', __name__)

supabase_client = SupabaseClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


@search_image.route('/')
def index():
    return render_template('index.html')


@search_image.route('/image-search', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"message": "File type not allowed"}), 400

    try:
        # Initialize model
        model = image_extract.get_extract_model()

        # Extract features for the search image
        search_vector = image_extract.extract_vector(model, file)

        # Load vectors and paths from files
        vectors = load_file(name.FILE_VECTORS)
        paths = load_file(name.FILE_PATHS)

        # Ensure vectors are in the correct shape
        vectors = np.atleast_2d(vectors)
        search_vector = np.atleast_2d(search_vector)

        # Load the FAISS index
        index = faiss.read_index(name.FILE_FAISS_INDEX)

        # Perform the search with FAISS to find the K nearest neighbors
        K = 12
        distances, indices = index.search(search_vector, K)

        # Retrieve the closest images and their distances
        nearest_images = [
            {"path": supabase_client.get_one_file_from_bucket_by_url(paths[idx], name.BUCKET_IMAGES),
             "distance": round(float(dist), 5)}
            for idx, dist in zip(indices[0], distances[0])
        ]

        return jsonify({"nearest_images": nearest_images}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
