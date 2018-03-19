from flask import Flask, render_template, Response, request
from flask import send_from_directory
from camera import VideoCamera
import os
import cv2
import shutil, sys
import numpy as np
from PIL import Image
import base64
import re
import cStringIO

app = Flask(__name__)
app.debug = True
# @app.route('/')
@app.route("/")
def index():
    return render_template('one.html')



# @app.route("/upload", methods=['POST'])
# def upload():
#     imagefile = request.files.get('imagefile', '')
#     return imagefile


@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(cStringIO.StringIO(image_b64))
    image_np = np.array(image_PIL)
    print 'Image received: {}'.format(image_np.shape)
    return ''

# @app.route('/', methods=['POST'])
# def upload():
#     try:
#         imagefile = flask.request.files.get('imagefile', '')
#     except Exception as e:
#         raise e
#     return render_template('boc.html')


def app_make():
    app.run(host='0.0.0.0',ssl_context=('cert.pem', 'key.pem'), debug=True)
    # app.run(host='0.0.0.0', debug=True)

app_make()