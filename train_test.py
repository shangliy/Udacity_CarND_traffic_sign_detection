import pickle
import math
import numpy as np
import tensorflow as tf
import time
import os.path
import sys

def normalization(image_data):
    return ((image_data)/255-0.5)*2

input_tensor = 'anchor:0'
output_tensor = 'fc_2/add:0'

with open('data/test.p', mode='rb') as f:
  test = pickle.load(f)
test_features = normalization(test['features'])
test_labels = test['labels']

######################################################################
# Unpersists graph from file
with tf.gfile.FastGFile("train.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
######################################################################
correct = 0
num = 0
i = 0
with tf.Session() as sess:
    out_tensor = sess.graph.get_tensor_by_name(output_tensor)
    inputs = np.zeros((100,32,32,3),dtype= "float32")
    labels = np.zeros((100,1),dtype= "int32")
    while i <12000:


        for j in range(100):
            inputs[j] = test_features[i+j,:,:,:]
            labels[j] = test_labels[i+j,]


        output = sess.run([out_tensor], {input_tensor: inputs})
        print(output[0].shape)
        predicts = np.argmax(output[0],1);

        print("predict_shape:"+str(predicts.shape))

        labels = test_labels[i:i+100,]


        correct = correct + np.sum(predicts == labels)
        num = num + 100
        i = i + 100
        print(np.sum(predicts == labels))
        


print (correct/num)
