import os

from flask import Flask
from flask_cors import CORS  # Import CORS

from routes.extract_image import extract_image as extract_image_blueprint
from routes.search_image import search_image as search_image_blueprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Enable CORS for the entire application
CORS(app)

# Register the Blueprints
app.register_blueprint(search_image_blueprint, url_prefix='/api')
app.register_blueprint(extract_image_blueprint, url_prefix='/api')

# @app.route('/')
# def index():
#     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
