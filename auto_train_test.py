from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from numpy import array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
from keras.models import model_from_yaml
from sklearn.model_selection import StratifiedKFold
import keras

def main(mode,classifier_filename, model_type):
    seed=102
    min_nrof_images_per_class=20
    nrof_train_images_per_class=40
    batch_size=1000
    image_size=160
    data_dir='out_dir_small' 
    use_split_dataset=True
    model_dir = './pre_model/20170511-185253.pb'
    classe_total = 4

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=seed)

            if use_split_dataset:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = split_dataset(dataset_tmp, min_nrof_images_per_class, nrof_train_images_per_class)
                if (mode=='TRAIN'):
                    dataset = train_set
                elif (mode=='TEST'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            # for cls in dataset:
                # assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')


            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            if (mode=='TRAIN'):
                # Train classifier
                if(model_type=='SVM'):
                    print('Training classifier')
                    model = SVC(kernel='linear', probability=True)
                    model.fit(emb_array, labels)

                    # Create a list of class names
                    class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                    # Saving classifier model
                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((model, class_names), outfile)
                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
                if(model_type=='NN'):
                    label_array = array(labels)
                    one_hot_labels = keras.utils.to_categorical(label_array, num_classes=classe_total)
                    print('Training classifier')


                    model = Sequential()
                    model.add(Dense(32, activation='relu', input_dim=128))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(classe_total, activation='softmax'))
                    model.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
                    model.fit(emb_array, one_hot_labels, epochs=100, batch_size=32)
                    # model.fit(emb_array, one_hot_labels, epochs=100, batch_size=32, validation_split=0.33)
                    print('trainging finished for NN Facenet Model')
                    scores = model.evaluate(emb_array, one_hot_labels, verbose=0)
                    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                    # serialize model to YAML
                    model_yaml = model.to_yaml()
                    with open("./my_class/model_small.yaml", 'wb') as yaml_file:
                        yaml_file.write(model_yaml)
                    model.save(classifier_filename)
                    print("Saved model to disk")
                    

            elif (mode=='TEST'):
                # Classify images
                if(model_type == "SVM"):
                    print('Testing classifier')
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    print('Loaded classifier model from file "%s"' % classifier_filename_exp)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    for i in range(len(best_class_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print('Accuracy: %.3f' % accuracy)
                if(model_type=="NN"):
                    label_array = array(labels)
                    one_hot_labels = keras.utils.to_categorical(label_array, num_classes=classe_total)
                    yaml_file = open('./my_class/model_small.yaml', 'r')
                    loaded_model_yaml = yaml_file.read()
                    yaml_file.close()
                    loaded_model = model_from_yaml(loaded_model_yaml)
                    # load weights into new model
                    loaded_model.load_weights(classifier_filename)
                    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                    print("Loaded model from disk")
                    score = loaded_model.evaluate(emb_array, one_hot_labels, verbose=0)
                    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
                

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set



def NN(option):
    if (option == 'TRAIN'):
        main('TRAIN', './my_class/my_model_small.h5', 'NN')
    else:
        main('TEST', './my_class/my_model_small.h5', 'NN')

def SVM():
    main('TRAIN', './my_class/my_classifier_small.pkl', 'SVM')
    main('TEST', './my_class/my_classifier_small.pkl', 'SVM')

if __name__ == '__main__':
    NN('TEST')
    # NN('TRAIN')
    # SVM()