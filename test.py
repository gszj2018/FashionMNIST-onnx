#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original: https://github.com/LC-John/Fashion-MNIST/blob/master/code/test.py
Change: Export ONNX model.
"""

import tf2onnx
import tensorflow.compat.v1 as tf
import numpy
import random
import time
import pickle
import os, sys
from fmnist_dataset import Fashion_MNIST
from model import CNN

tf.app.flags.DEFINE_integer("rand_seed", 2019,
                            "seed for random number generaters")
tf.app.flags.DEFINE_string("gpu", "0",
                           "select one gpu")

tf.app.flags.DEFINE_integer("n_correct", 1000,
                            "correct example number")
tf.app.flags.DEFINE_string("correct_path", "../attack_data/correct_1k.pkl",
                           "pickle file to store the correct labeled examples")
tf.app.flags.DEFINE_string("model_path", "../model/fmnist_cnn.ckpt",
                           "check point path, where the model is saved")

tf.app.flags.DEFINE_string("dtype", "fp32",
                           "data type. \"fp16\", \"fp32\" or \"fp64\" only")

flags = tf.app.flags.FLAGS

if __name__ == "__main__":
    tf.disable_eager_execution()
    
    print ("[*] Hello world!", flush=True)
    
    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    
    # Set random seed
    tf.set_random_seed(flags.rand_seed)
    random.seed(flags.rand_seed)
    numpy.random.seed(flags.rand_seed)
    
    # Load dataset
    d = Fashion_MNIST()
    
    # Read hyper-parameters
    n_correct = flags.n_correct
    correct_path = flags.correct_path
    model_path = flags.model_path
    if flags.dtype == "fp16":
        dtype = numpy.float16
    elif flags.dtype == "fp32":
        dtype = numpy.float32
    elif flags.dtype == "fp64":
        dtype = numpy.float64
    else:
        assert False, "Invalid data type (%s). Use \"fp16\", \"fp32\" or \"fp64\" only" % flags.dtype
    
    # Build model
    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", is_inference=True)
        print("[*] Model built!")
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        m.restore(sess, model_path)
        
        print("[*] Model loaded!")
        print("[*] Model parameters:")
        parm_cnt = 0
        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print("   ", v.name, v.get_shape())
            parm_cnt_v = 1
            for i in v.get_shape().as_list():
                parm_cnt_v *= i
            parm_cnt += parm_cnt_v
        print("[*] Model parameter size: %.4fM" %(parm_cnt/1024/1024))

        print(m.XXX.name)
        print(m.YYY.name)

        g = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [m.YYY.name.split(':')[0]])
        model_proto, stor = tf2onnx.convert.from_graph_def(g, "blackbox", input_names=[m.XXX.name], output_names=[m.YYY.name])
        with open("model.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())
        
