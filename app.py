from flask import Flask, jsonify, request,render_template
from flask import Blueprint, redirect, url_for,  flash
from flask import request

# from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os

import cv2, numpy
import time

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage



app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = "uploads"

@app.route('/', methods=['GET', 'POST'])
def get_image():
    print (509)
    # model_inference()
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    #read image file string data

    filestr = request.files['photo'].read()
    #convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    print (img.shape)
    cv2.imwrite('static/img.jpg', img)
    print (f'shape of the image is {img.shape}')

    # print (filestr.)

    #inference the model
    print ('starting inference')
    st = time.time()
    output = model_inference()
    print (time.time() - st)
    if type(output)==str:
        print ('nothing found')

    print (os.getcwd())
    if os.getcwd().endswith('5'):
        os.chdir('..')

    print (output)


    img_resized = cv2.resize(img, (350,350))
    # img_resized = img
    print (img_resized.shape)

    x1 = (float(output['x_centre']) - float(output['width']) / 2) * img_resized.shape[0]
    x2 = (float(output['x_centre']) + float(output['width']) / 2) * img_resized.shape[0]
    y1 = (float(output['y_centre']) - float(output['height']) / 2) * img_resized.shape[1]
    y2 = (float(output['y_centre']) + float(output['height']) / 2) * img_resized.shape[1]
    

    print (x1, x2, y1, y2)


    IM_PATH = 'static/img_resized.jpg'
    cv2.imwrite(IM_PATH, img_resized)
    print ('saved resized')

    # time.sleep(4) 
    IM_PATH_BBOX = 'static/img_bbox.jpg' 
    print ('drawing bbox')
    img_bbox = draw_bboxes(img_resized, x1=x1, x2=x2, y1=y1, y2=y2, color=[(247, 121,64), (255, 0,255), (255, 255, 0)][int(output['class'])])
    cv2.imwrite(IM_PATH_BBOX, img_bbox)
    print ('saved image with_bbox')


    IM_PATH_DEFECT_CLOSE = 'static/defect_close.jpg'
    print (f'im bbox {img_bbox.shape}')
    e = 0
    defect_close = img_resized[int(y1)-e : int(y2)+e, int(x1)-e : int(x2)+e]
    print (f'defect_close {defect_close.shape}')
    cv2.imwrite(IM_PATH_DEFECT_CLOSE, defect_close)


    IM_LAPLACE_PATH = 'static/img_laplacian.jpg'
    im_laplace = cv2.Laplacian(defect_close,cv2.CV_64F)
    cv2.imwrite(IM_LAPLACE_PATH, im_laplace)

    # print ('start sleeping')
    # time.sleep(10)
    # return f"<img src='C:/Users/User/Desktop/Hachathon/App/templates/img.jpg'>"
    return render_template('upload_2.html', class_=['Трещина', 'Недолив' ,'Раковина'][int(output['class'])], conf=numpy.round(100 * float(output['confidence']), 2), \
                        image_path=IM_PATH,  \
                        image_path_bbox=IM_PATH_BBOX, im_laplace=IM_LAPLACE_PATH, defect_close_path = IM_PATH_DEFECT_CLOSE)
   

def draw_bboxes(image, x1=None, x2=None, y1=None, y2=None, label="Раковина", color=(247, 121,64)):

    bbs = BoundingBoxesOnImage([
                BoundingBox(x1=x1, x2=x2, \
                            y1=y1, y2=y2, label=label)
                                ], shape=image.shape)

    image = bbs.draw_on_image(image, size=2, color=color)

    return image

def model_inference():
    confidence = 0.25
    if os.getcwd().endswith('App'):
        os.chdir('yolov5')
    # print ('current directory is')
    # print (os.system('cd'))                                                   # weights/best.pt
    os.system(f'python detect.py --source C:/Users/User/Desktop/Hachathon/App/static/img.jpg --weights weights/best.pt --conf {confidence} --save-txt --save-conf --save-crop')


    last_exp = newest_file('runs//detect')

    last_exp = last_exp.split("\\",1)[1]
    output = []
    try:
        with open(f'runs//detect//{last_exp}//labels//img.txt') as f:
            output = f.read().split(" ")
        print (output)
        out = {}
        out['class'] = output[0]
        out['x_centre'] = output[1]
        out['y_centre'] = output[2]
        out['width'] = output[3]
        out['height'] = output[4]
        out['confidence'] = output[-1]
        print (out)
        return out

    except Exception as e:
        print (e)
        confidence = 0.25 // 3
        if os.getcwd().endswith('App'):
            os.chdir('yolov5')
        # print ('current directory is')
        # print (os.system('cd'))                                                   # weights/best.pt
        os.system(f'python detect.py --source C:/Users/User/Desktop/Hachathon/App/static/img.jpg --weights weights/aug_best.pt --conf {confidence} --save-txt --save-conf --save-crop')


        last_exp = newest_file('runs//detect')

        last_exp = last_exp.split("\\",1)[1]
        output = []
        try:
            with open(f'runs//detect//{last_exp}//labels//img.txt') as f:
                output = f.read().split(" ")
            print (output)
            out = {}
            out['class'] = output[0]
            out['x_centre'] = output[1]
            out['y_centre'] = output[2]
            out['width'] = output[3]
            out['height'] = output[4]
            out['confidence'] = output[-1]
            print (out)
            return out
        except Exception as e:
            print (e)
            return "nothing found"
    # pass

def newest_file(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)



# multiple uploads    
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'asdsdfdfgshdfsd'
# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/add_defect')
def upload_form():
    return render_template('add_defect.html')

@app.route('/add_defect', methods=['GET',"POST"])
def upload_form_post():
    print (request.form)
    name = request.form['label']
    cords = request.form['cords']

    print (name, cords)

    return render_template('add_defect.html')



# @app.route('/add_defect', methods=['POST'])
# def upload_file():
#     if request.method == 'POST':
#         print ('starting')
#         if 'files[]' not in request.files:
#             flash('No file part')
#             return redirect(request.url)

#         files = request.files.getlist('files[]')

#         for file in files:
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#         flash('File(s) successfully uploaded')
#         return redirect('/add_defect')



# @app.route('/add_defect', methods=['GET', 'POST'])
# def hello():
#     print ('st')
#     # POST request
#     if request.method == 'POST':
#         print('Incoming..')
#         print(request.get_json())  # parse as JSON
#         return 'OK', 200
#     return str(request)


if __name__ == '__main__':
    app.debug = True
    # app.run(host='0.0.0.0', port=3000)
    app.run(use_reloader=True,host='127.0.0.1', port=5001)
