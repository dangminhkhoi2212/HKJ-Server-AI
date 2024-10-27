from dotenv import load_dotenv
from flask import jsonify, request, Blueprint

from lib.supabase_client import SupabaseClient
from services.image_extract_service import ImageExtractService
from services.image_serach_service import ImageSearchSystem
from services.jewelry_image_vector_service import JewelryImageVectorService

# loading variables from .env file

load_dotenv()
extract_image = Blueprint('extract_image', __name__)

supabase_client = SupabaseClient()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

image_search_system = ImageSearchSystem()

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
