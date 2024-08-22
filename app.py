# app.py
import os

from flask import Flask

from routes.search_image import search_image as search_image_blueprint
from routes.train import train as train_blueprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Register the Blueprint
app.register_blueprint(search_image_blueprint)
app.register_blueprint(train_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
