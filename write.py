import os
import numpy as np
import caffe
import csv


mean_file = 'ilsvrc_2012_mean.npy'
deploy_path = 'sentiment_deploy.prototxt'
caffemodel_path = 'twitter_finetuned_test4_iter_180.caffemodel'

net = caffe.Classifier(deploy_path,
                       caffemodel_path,
                       mean=np.load(mean_file).mean(1).mean(1),
                       image_dims=(256, 256),
                       channel_swap=(2, 1, 0),
                       raw_scale=255)

output_csv = 'output.csv'
target_folder = './ferrero/'
i = 0
with open(output_csv, 'wb') as output_file:
    row_writer = csv.writer(output_file, delimiter=',')
    for image_path in os.listdir(target_folder):
        im = caffe.io.load_image(str(target_folder + image_path))
        prediction = net.predict([im], oversample=False)
        row_writer.writerow([image_path, prediction[0].argmax()])
        print i 
        i +=1




# # Make a forward pass and get the score
#
# print prediction
# print prediction[0].argmax()

