
import json

from flask import Flask,jsonify,request
# from yolo_detection_images_2 import detectObjects
from yolo_detection_images import detectObjects
# from yolo_detection_images_gray import detectObjects
from flask_ngrok import run_with_ngrok




#app=Flask(__name__)
app = Flask(__name__, static_url_path = "", static_folder = "D:/test_flask/YOLO-v3-Object-Detection")
app.config['JSON_AS_ASCII'] = False
# import os
# from flask import send_from_directory

# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                                'favicon.ico', mimetype='images/favicon.ico')


@app.route('/')
def detect():
    img=request.args['image']
    # img_path='images/'+img
    img_path=img
    results=detectObjects(img_path)
    # return jsonify(results)
    return json.dumps(results,ensure_ascii=False)

run_with_ngrok(app)
if __name__ == "__main__":
    app.run()