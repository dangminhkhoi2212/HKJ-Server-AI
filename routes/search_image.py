# modules/routes.py

from flask import Blueprint, request, jsonify

from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.image_serach_service2 import ImageSearchService

# Create a Blueprint for the routes
search_image = Blueprint('search_image', __name__)

supabase_client = SupabaseClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

image_search_system = ImageSearchService()
image_extract_service = ImageExtractService()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


def to_camel_case(snake_str):
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def convert_keys_to_camel_case(data):
    """Convert all keys in a dictionary to camelCase."""
    if isinstance(data, dict):
        return {to_camel_case(k): convert_keys_to_camel_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_camel_case(i) for i in data]
    else:
        return data


@search_image.route('/search-image', methods=['POST'])
async def search_images():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files['file']
        print(f'Input file:{file}')
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"message": "File type not allowed"}), 400
            # Convert the uploaded file to a PIL image\
        file_bytes = file.read()
        img_no_bg = image_extract_service.remove_bg_image(file_bytes)
        if img_no_bg is None:
            return jsonify({"message": "Remove background image error"}), 400

        nearest_images = image_search_system.search_image(img_no_bg)
        camel_case_response = convert_keys_to_camel_case(nearest_images)
        return jsonify(camel_case_response), 200

    except Exception as e:
        print(f"Error searching image: {str(e)}")
        return jsonify({'search image error': str(e)}), 500
