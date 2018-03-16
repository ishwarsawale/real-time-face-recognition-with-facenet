
# coding: utf-8

# In[7]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import keras
from numpy import array


# In[11]:
def train_nn():
    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = './out_dir'
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print (dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            print('Loading feature extraction model')
            modeldir = './pre_model/20170511-185253.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 1000
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                print('getting embbeding for images : ', i)
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            # classifier_filename = './my_class/my_classifier.pkl'
            # classifier_filename_exp = os.path.expanduser(classifier_filename)
            label_array = array(labels)
            print (emb_array.shape)
    #         print(labels)
    #         print(type(label_array))
            print(label_array.shape)
            one_hot_labels = keras.utils.to_categorical(label_array, num_classes=4)
            print(one_hot_labels.shape)
            print(one_hot_labels)
            # Train classifier
            print('Training classifier')
            # model = SVC(kernel='sigmoid', probability=True)
            # model.fit(emb_array, labels)

            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=128))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(4, activation='softmax'))
            model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(emb_array, one_hot_labels, epochs=100, batch_size=32)
            model.save('my_model.h5')
            print('trainging finished for NN Facenet Model')
# train_nn()