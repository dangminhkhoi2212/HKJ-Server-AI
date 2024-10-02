import os

from flask import Flask
from flask_cors import CORS  # Import CORS

from routes.image import image as image_blueprint
from routes.search_image import search_image as search_image_blueprint
from routes.train import train as train_blueprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Enable CORS for the entire application
CORS(app)

# Register the Blueprints
app.register_blueprint(search_image_blueprint, url_prefix='/api')
app.register_blueprint(train_blueprint, url_prefix='/api')
app.register_blueprint(image_blueprint, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')
