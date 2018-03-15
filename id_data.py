import os
import detect_and_align
from scipy import misc
import numpy as np
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

class ID_Data():
    def __init__(self, name, image_path):
        self.name = name
        self.image_path = image_path
        self.embedding = []


def get_id_data():
    id_dataset = []
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
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            id_folder = datadir
            ids = os.listdir(os.path.expanduser(id_folder))
            ids.sort()
            for id_name in ids:
                print(id_name)
                id_dir = os.path.join(id_folder, id_name)
                image_names = os.listdir(id_dir)
                image_paths = [os.path.join(id_dir, img) for img in image_names]
                for image_path in image_paths:
                    # print(image_paths)
                    id_dataset.append(ID_Data(id_name, image_path))

    # aligned_images = align_id_dataset(id_dataset, pnet, rnet, onet)

    # feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
    # emb = sess.run(embeddings, feed_dict=feed_dict)
            print(len(id_dataset))
            for i in range(len(id_dataset)):
                id_dataset[i].embedding = emb_array[i, :]
    return id_dataset


def align_id_dataset(id_dataset, pnet, rnet, onet):
    aligned_images = []

    for i in range(len(id_dataset)):
        image = misc.imread(os.path.expanduser(id_dataset[i].image_path), mode='RGB')
        face_patches, _, _ = detect_and_align.align_image(image, pnet, rnet, onet)
        aligned_images = aligned_images + face_patches

    aligned_images = np.stack(aligned_images)
    return aligned_images


if __name__ == "__main__":
    main()
