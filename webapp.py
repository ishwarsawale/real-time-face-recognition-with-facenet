
from flask import Flask, render_template, Response, request
from camera import VideoCamera
import os
import cv2
app = Flask(__name__)
app.debug = True
# @app.route('/')
@app.route("/")
def index():
    return render_template('boc.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    f_name = request.form.get('first_name')
    l_name = request.form.get('last_name')
    handle_data.path = f_name +" "+ l_name
    if not os.path.exists(handle_data.path):
      os.makedirs(handle_data.path)
    return render_template("index.html")

@app.route('/thanks')
def store_data():
    print('store_data')
    # sourcepath='./'
    # sourcefiles = os.listdir(sourcepath)
    # destinationpath = handle_data.path
    # print('called me')
    # for file in sourcefiles:
    #     if file.endswith('.jpg'):
    #         shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))
    return render_template("thanks.html")
def gen(camera):
    while True:
        frame = camera.get_frame()
        image = camera.get_image()
        global count
        count += 1
        # print(count)
        if count <= 500:
          cv2.imwrite("%s_%d.jpg" % (handle_data.path, count), image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
count = 0
@app.route('/video_feed', methods=['GET'])
# @app.route('/video_feed')
def video_feed():

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)