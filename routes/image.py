from flask import jsonify, request, Blueprint

from constants.name import BUCKET_IMAGES
from lib.supabase_client import SupabaseClient
from routes.search_image import allowed_file

image = Blueprint('image', __name__)

supabase_client = SupabaseClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@image.route('/upload-image', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "File type not allowed"}), 400

        if request.content_length > MAX_FILE_SIZE:
            return jsonify({"success": False, "message": "File size exceeds limit"}), 400

        response = supabase_client.upload_bucket(file, 'images')

        return jsonify({"success": True, "message": "File uploaded successfully", "url": str(response)}), 200

    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e)}), 404
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"An unexpected error occurred: {str(e)}"}), 500


@image.route('/delete-images', methods=['POST'])
def delete():
    try:
        if 'files' not in request.json:
            return jsonify({"success": False, "message": "No files provided"}), 400

        files = request.json['files']
        list_files = [f'{path.split("/")[-1]}' for path in files]
        print(list_files)
        response = supabase_client.delete_files_from_bucket(BUCKET_IMAGES, list_files)
        return jsonify({"success": True, "message": "Files deleted successfully", "response": response}), 200
    except Exception as e:
        return jsonify({"success": False, "message": f"An unexpected error occurred: {str(e)}"}), 500
