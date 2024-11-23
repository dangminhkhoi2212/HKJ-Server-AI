from dotenv import load_dotenv
from flask import jsonify, request, Blueprint

from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.jewelry_image_vector_service import JewelryImageVectorService

# loading variables from .env file

load_dotenv()
extract_image = Blueprint('extract_image', __name__)

supabase_client = SupabaseClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

image_extract_service = ImageExtractService()
jewelry_image_vector_service = JewelryImageVectorService()


# @extract_image.before_request
# async def init_search_system():
#     await search_system.init_system()
#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


@extract_image.route('/extract-image', methods=['POST'])
async def add_image():
    try:
        data = request.get_json()
        images = data.get('images')
        print(images)
        result = image_extract_service.process_image(images)
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@extract_image.route('/extract-image/remove', methods=['POST'])
async def remove_image():
    try:
        data = request.get_json()
        images = data.get('images')
        print(f'images remove: {images}')
        result = jewelry_image_vector_service.remove_images(images)
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@extract_image.route('/extract-image/test/remove-bg', methods=['POST'])
async def remove_bg_image():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files['file']
        print(f'Input file:{file}')
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"message": "File type not allowed"}), 400

        image_extract_service.remove_bg_image(file.read())
        return jsonify({'success': True, })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
