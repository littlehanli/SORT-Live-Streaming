# python object_tracker.py --ip 0.0.0.0 --port 8000
from models import *
from utils import *
from sort import *
import cv2

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

from imutils.video import VideoStream
from flask import Flask, render_template, Response, jsonify,request
import threading, argparse, imutils

outputFrame = None
lock = threading.Lock()
all_id = []
get_data = -1

# initialize a flask object
app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
    

def mouse_click(event, x, y, flags, param):
    global mouseX, mouseY, select

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        print('left: ', x, y)

    if event == cv2.EVENT_RBUTTONDOWN:
        select = -1
        mouseX, mouseY = -1, -1
        print('Right: Canceled ', select)


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def detect_motion():
    # lock variables
    global vs, outputFrame, lock, all_id

    videopath = 'data/video01.mp4'

    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    #vid = cv2.VideoCapture(videopath)
    #vid = cv2.VideoCapture(0)
    mot_tracker = Sort() 

    #cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Stream', (800,600))

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #ret,frame=vid.read()
    #vw = frame.shape[1]
    #vh = frame.shape[0]
    #print ("Video size", vw,vh)
    #outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

    frames = 0
    starttime = time.time()
    mouseX, mouseY = -1, -1
    select = -1
    while(True):
        #ret, frame = vid.read()
        #if not ret:
        #    break
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        frames += 1
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            all_id = tracked_objects[:,4]
            for i,  value in enumerate(tracked_objects):
                x1, y1, x2, y2, obj_id, cls_pred = value
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                if obj_id == return_get_data() or return_get_data() == -1:
                    # print('---------- ',select,' ----------')
                    color = colors[int(obj_id) % len(colors)]
                    cls = classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        cv2.imshow('Stream', frame)
        #outvideo.write(frame)
        
        cv2.setMouseCallback('Stream', mouse_click)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 32:
            print("Cancel selection")
            select = -1

        if ch == 27:
            break
        
        with lock:
            outputFrame = frame.copy()
  
    totaltime = time.time()-starttime
    print(frames, "frames", totaltime/frames, "s/frame")
    #cv2.destroyAllWindows()
    #outvideo.release()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/tracking_list',methods=['GET'])
def tracking_list():
    #data = [random.randrange(1, 10, 1) for i in range(7)]
    data = all_id.tolist()
    print("---tracking_list", len(data), data)
    if data is not None:
        data.insert(0,'All')
        data.insert(-1,'None')
    else:
        data = ['All','None']
    print(data)
    #return str(datetime.datetime.now())  # 示範用
    return jsonify(data)

@app.route('/get_select_id',methods=['GET','POST'])
def get_select_id():
    global get_data
    # send data to js
    if request.method == 'GET':
        print("---get_select_id: ",get_data)
        return str(get_data)

    # receive data from js and return 
    elif request.method == 'POST':
        print("---post_select_id: ", request.values['id'])
        get_data = request.values['id']
        if get_data is not 0:
            return jsonify(dict(id=get_data,)), 201

def return_get_data():
    if get_data == 'None':
        return -2
    elif get_data == 'All':
        return -1
    else:
        return int(get_data)

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
    
# release the video stream pointer
vs.stop()
