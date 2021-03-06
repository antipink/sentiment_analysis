from flask import Flask, render_template, request, url_for, jsonify, make_response
from flask_cors import CORS, cross_origin
from PIL import Image
from subprocess import check_output
import caffe
import cv2
import numpy as np
from io import BytesIO
import re
import base64
from PIL import Image
import uuid

app = Flask(__name__)
app.debug = True

MAX_FILE_SIZE = 16 #in Mb
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE * 1024 * 1024

CORS( app)

@app.route("/")
def index():
    return ""

#CMD_TEMPLATE = "th run_model.lua -input_image {image_filename}"

mean_file = 'ilsvrc_2012_mean.npy'
deploy_path = 'sentiment_deploy.prototxt'
caffemodel_path = 'twitter_finetuned_test4_iter_180.caffemodel'



# Load network
net = caffe.Classifier(deploy_path,
                           caffemodel_path,
                           mean=np.load(mean_file).mean(1).mean(1),
                           image_dims=(256, 256),
                           channel_swap=(2, 1, 0),
                           raw_scale=255)


@app.route('/upload', methods=["GET","POST"])
def receive():
# try:
#     if request.method == 'POST':
        print(request.json, request.args, request.files)
        # files = list(request.files.values())
        # input_image = files[0]
        # im = caffe.io.load_image(input_image.stream)

        args = request.get_json()

        image_str = args["image"]

        image_data = re.sub('^data:image/.+;base64,', '', image_str)
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        # image_data = decodestring(imagestr)
        filename = str(uuid.uuid4()) + ".jpg"
        im.save(filename)

        # with open(filename, "rb") as imagefile: 
        print(filename)
        im = caffe.io.load_image(filename)
            #im = Image.open(imagefile)
        prediction = net.predict([im], oversample=False)

        #file = request.files['photo']
        #im = Image.open(file.stream)
        #assuming only one file is sent

        # print( request.files)
        # print( input_image)
        # img_filename = "temp.jpg"

        #im = Image.open(input_image.stream)
        #im.save(img_filename, 'JPG')
        # input_image.save(img_filename)

        # Load image
        # im = caffe.io.load_image(image_path)

        # Make a forward pass and get the score
        # prediction = net.predict([im], oversample=False)


        print prediction[0].argmax()
        if prediction[0].argmax() == 1:
           result = 'Postive'
        else:
           result = 'Negative'

        dic = {'sentiment': result}

        # command = CMD_TEMPLATE.format( image_filename=img_filename)

        # print(command)
        # captioner_return = check_output(command, shell=True)

        return jsonify(dic)
#except:
#   return ""

@app.route('/test_api')
def helloworld():
    return "Hello world!"


if __name__ == "__main__":
    app.run("0.0.0.0", port=8888)
