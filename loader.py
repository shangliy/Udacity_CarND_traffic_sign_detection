import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

grap_path = "./train.pb"
with tf.Session() as sess:
    print("Load Graph:" + grap_path)
    with gfile.FastGFile(grap_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #x= tf.import_graph_def(graph_def, return_elements=["DecodeJpeg"])
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name ="")
    for op in sess.graph.get_operations():
        print (op.name)
    #print("map variables")
    #print sess.graph.get_operation_by_name('DecodeJpeg')
    #print sess.graph.get_operation_by_name('final_result')
    #print sess.graph.get_variable_by_name('DecodeJpeg:0')
    #print sess.graph.get_variable_by_name('final_result:0')
    #print (sess.graph.get_operations())
    summary_writer = tf.train.SummaryWriter('./graph/logs', graph=sess.graph)
   # test = tf.get_default_graph().get_operation_by_name("19_fc")
    #print test

#tensorboard --logdir=./work/logs
