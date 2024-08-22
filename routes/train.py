from io import BytesIO

import requests
from PIL import Image
from flask import Blueprint, request, jsonify
from matplotlib import pyplot as plt

from helpers import image_extract

train = Blueprint('train', __name__)


@train.route('/train', methods=['POST'])
def index():
    data = request.get_json()
    img_url = data.get('imgUrl')
    try:
        # Fetch the image from the URL
        response = requests.get(img_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        img = Image.open(BytesIO(response.content))

        # Display the image using matplotlib
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.show()

        # Convert PIL Image to BytesIO object for extraction
        img_bytes = BytesIO()
        img.save(img_bytes, format=img.format)
        img_bytes.seek(0)  # Rewind the BytesIO object to the beginning

        # Extract vector from the image
        model_extract = image_extract.get_extract_model()
        image_vector = image_extract.extract_vector(model_extract, img_bytes)

        # Return the image vector
        return jsonify({"image_vector": image_vector.tolist()})  # Convert numpy array to list for JSON serialization

    except Exception as e:
        return jsonify({"message": f"Error fetching or displaying the image: {str(e)}"}), 500
