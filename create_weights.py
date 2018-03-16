
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def facenet_128D():
    print('Creating networks and loading parameters')
    id_dataset = id_data.get_id_data()
    with open('weights', 'wb') as fp:
        pickle.dump(id_dataset, fp)

# facenet_128D()