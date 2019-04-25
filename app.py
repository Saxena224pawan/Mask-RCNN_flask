from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import io
import skimage.io
# Kerasfrom io import StringIO
import base64
from urllib.parse import quote
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from coco import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')
# Local path to trained weights file
ROOT_DIR = os.path.abspath("../")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Download COCO trained weights from Releases if needed
# Create model object in inference mode.
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

def model_predict(img_path, model):
    
    img1=skimage.io.imread(img_path)
    # Preprocessing the image
    results = model.detect([img1]   , verbose=1)
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    # Visualize results
    r = results[0]
    return visualize.display_instances(img1, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

file_path=""
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == "POST":
        fig = plt.figure()
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        preds = model_predict(file_path,model)
        img = io.BytesIO()
        preds.savefig(img, format='png',bbox_inches="tight")
        img.seek(0)
        data = base64.b64encode(img.getvalue()).decode()
        data_url = 'data:image/png;base64,{}'.format(quote(data))
        return data_url
    return None
"""       
@app.route('/prediction', methods=['GET'])
def gred():
    print(request.args.get(file_path))
    return render_template("im1.html",url = 'static/new_plot.jpg',url1= str(request.args.get(file_path)))
"""    
if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
