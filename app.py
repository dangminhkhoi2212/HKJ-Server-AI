import os
import time

from flask import Flask
from flask_apscheduler import APScheduler
from flask_cors import CORS  # Import CORS

from routes.extract_image import extract_image as extract_image_blueprint
from routes.search_image import search_image as search_image_blueprint
from services.build_faiss_service import BuildFaissService

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
app.config['SCHEDULER_API_ENABLED'] = True

scheduler = APScheduler()
# Enable CORS for the entire application
CORS(app)

buidl_faiss_service = BuildFaissService()


def scheduled_task():
    print(f"Scheduled task running at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # buidl_faiss_service.build_faiss_index()


# Khởi tạo và cấu hình scheduler
scheduler.init_app(app)
scheduler.start()

# Thêm tác vụ vào lịch
scheduler.add_job(id='Scheduled Task', func=scheduled_task, trigger='cron',
                  hour=0,
                  minute=0,
                  )

# Register the Blueprints
app.register_blueprint(search_image_blueprint, url_prefix='/api')
app.register_blueprint(extract_image_blueprint, url_prefix='/api')

# @app.route('/')
# def index():
#     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
