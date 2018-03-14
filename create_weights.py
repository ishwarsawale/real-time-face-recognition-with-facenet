
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import pymongo
from pymongo import MongoClient
import datetime
from keras.models import load_model
import keras
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import detect_and_align
import id_data
from scipy import misc
import re
import cv2
import argparse
import time
import twoFace
import pickle



def facenet_128D():
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy')
            print('Loading feature extraction model')
            modeldir = './pre_model/20170511-185253.pb'
            facenet.load_model(modeldir)
            print('loaded model')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print('loaded graph')
            embedding_size = embeddings.get_shape()[1]
            id_dataset = id_data.get_id_data('./out_dir', pnet, rnet, onet, sess, embeddings, images_placeholder, phase_train_placeholder)
            with open('weights', 'wb') as fp:
                pickle.dump(id_dataset, fp)
            
facenet_128D()