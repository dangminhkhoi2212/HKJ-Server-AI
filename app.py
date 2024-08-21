# app.py
from flask import Flask
from routes.index import main as main_blueprint

app = Flask(__name__)

# Register the Blueprint
app.register_blueprint(main_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
