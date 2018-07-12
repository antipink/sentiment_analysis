
import os
import sys
import numpy as np
import caffe
import cv2
import argparse




subsets = ['test4']
mean_file = 'ilsvrc_2012_mean.npy'

output_path = './result/'

image_path = '1.jpg'
# Update paths for this subset
deploy_path = 'sentiment_fully_conv_deploy.prototxt'
caffemodel_path = 'twitter_finetuned_test4_iter_180_conv.caffemodel'

# Load network
net_full_conv = caffe.Net(deploy_path, caffemodel_path, caffe.TEST)

# Configure preprocessing
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)

# Load image
im = caffe.io.load_image(image_path)

# Make a forward pass
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out
image_name = image_path.split('/')[-1]
np.save('./' + image_name.split('.', 2)[0], out['prob'][0])
out = out['prob'][0]

heatmap = np.zeros((out.shape[1], out.shape[2], 3))  # BGR
heatmap[:, :, 1] = 255 * out[1]  # positive (1) in green
heatmap[:, :, 2] = 255 * out[0]  # negative (0) in red

im = cv2.imread(image_path)

heatmap = cv2.resize(heatmap, tuple(im.shape[1::-1]), interpolation=cv2.INTER_NEAREST)

# Combine image and heatmap
output = 0.5*im + 0.5*heatmap

cv2.imwrite(os.path.join(output_path,image_name), output)
