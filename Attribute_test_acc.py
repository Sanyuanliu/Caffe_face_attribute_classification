#coding: utf-8

import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# this file should be run from {caffe_root}
sys.path.append('./python') 
import caffe
attribute_acceracy = np.zeros(40)


eval_model_def = '/home/sanyuan/CaffeProject/FA_v1-attention/models/attentionRoi/deploy.prototxt'
eval_model_weights = '/home/sanyuan/CaffeProject/FA_v1-attention/models/attentionRoi/alexnet_iter_70000.caffemodel'

#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(eval_model_def, eval_model_weights, caffe.TEST)
input_shape = net.blobs['data'].data.shape
sample_shape = net.blobs['data'].data.shape

# Set transformer
transformer = caffe.io.Transformer({'data': input_shape})
transformer.set_transpose('data', (2,0,1)) #from 256 256 3 to 3* 256 *256 
transformer.set_mean('data', np.array([90.1146, 103.035, 127.689])) #or load the mean image

	
with open("/home/sanyuan/CaffeProject/FA_v1-attention/data/Attribute/val_crop.txt") as Test_list:
	lines = Test_list.readlines()
	for line in lines:
		img_name = line.split()[0]
		image_path = "/media/sanyuan/Sanyuan/Dataset/crop_by_4/"+img_name
		eval_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		#plt.figure()
		#plt.imshow(eval_image)
		#plt.show()
		rs_image = cv2.resize(eval_image, (input_shape[2],input_shape[3]))

		data = transformer.preprocess('data', rs_image)
		data=data*0.0078125
		xdata = np.zeros((1,3,256,256))
		xdata[0,...] = data
		out = net.forward(data=xdata)
		prob_data = net.blobs['prob_data']
		attribute_data =prob_data.data[0][0]
		for j in range(0,40):
			if(((attribute_data[j][0]>0.5)and(line.split()[j+1]=='0'))or((attribute_data[j][0]<0.5)and(line.split()[j+1]=='1'))):
				attribute_acceracy[j]+=1
for j in range(0,40):
	print (attribute_acceracy[j]/20258.0)

print "average acc:"
print (sum(attribute_acceracy)/(40*20258.0))



