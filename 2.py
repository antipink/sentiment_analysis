
import os
import sys
import numpy as np
import caffe
import cv2
import argparse



mean_file = 'ilsvrc_2012_mean.npy'

output_path = './result/'
image_path = '12.jpg'

# Update paths for this subset
deploy_path = 'sentiment_deploy.prototxt'
caffemodel_path = 'twitter_finetuned_test4_iter_180.caffemodel'

# Load network
net = caffe.Classifier(deploy_path,
                       caffemodel_path,
                       mean=np.load(mean_file).mean(1).mean(1),
                       image_dims=(256, 256),
                       channel_swap=(2, 1, 0),
                       raw_scale=255)

# Load image
im = caffe.io.load_image(image_path)

# Make a forward pass and get the score
prediction = net.predict([im], oversample=False)
print prediction
print prediction[0].argmax()
