from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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


def find_matching_id(id_dataset, embedding):
    threshold = 1.10
    min_dist = 7.80
    matching_id = None
    for id_data in id_dataset:
        dist = get_embedding_distance(id_data.embedding, embedding)
        if dist <= threshold: #and dist < min_dist:
            min_dist = dist
            matching_id = id_data.name
    return matching_id, min_dist

# def find_matching_id(id_dataset, embedding):
#     # threshold = 1.10
#     smallest = sys.maxsize
#     min_dist = 0.80
#     matching_id = None
#     thres = 0.6 
#     percent_thres = 70
#     for id_data in id_dataset:
#         dist = get_embedding_distance(id_data.embedding, embedding)
#         if (dist < smallest): #and dist < min_dist:
#             smallest = dist
#             matching_id = id_data.name
#     percentage =  min(100, 100 * thres / smallest)
#     if percentage <= percent_thres :
#         matching_id = "Unknown"
#     return matching_id, smallest


def get_embedding_distance(emb1, emb2):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    return dist


def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        print('restore graph')
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def print_id_dataset_table(id_dataset):
    nrof_samples = len(id_dataset)

    print('Images:')
    for i in range(nrof_samples):
        print('%1d: %s' % (i, id_dataset[i].image_path))
    print('')

    print('Distance matrix')
    print('         ', end='')
    for i in range(nrof_samples):
        name = os.path.splitext(os.path.basename(id_dataset[i].name))[0]
        print('     %s   ' % name, end='')
    print('')
    for i in range(nrof_samples):
        name = os.path.splitext(os.path.basename(id_dataset[i].name))[0]
        print('%s       ' % name, end='')
        for j in range(nrof_samples):
            dist = get_embedding_distance(id_dataset[i].embedding, id_dataset[j].embedding)
            print('  %1.4f      ' % dist, end='')
        print('')


def test_run(pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder, embeddings, id_dataset, test_folder):
    if test_folder is None:
        return

    image_names = os.listdir(os.path.expanduser(test_folder))
    image_paths = [os.path.join(test_folder, img) for img in image_names]
    nrof_images = len(image_names)
    aligned_images = []
    aligned_image_paths = []

    for i in range(nrof_images):
        image = misc.imread(image_paths[i])
        face_patches, _, _ = detect_and_align.align_image(image, pnet, rnet, onet)
        aligned_images = aligned_images + face_patches
        aligned_image_paths = aligned_image_paths + [image_paths[i]] * len(face_patches)

    aligned_images = np.stack(aligned_images)

    feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)

    for i in range(len(embs)):
        misc.imsave('outfile' + str(i) + '.jpg', aligned_images[i])
        matching_id, dist = find_matching_id(id_dataset, embs[i, :])
        if matching_id:
            print('Found match %s for %s! Distance: %1.4f' % (matching_id, aligned_image_paths[i], dist))
        else:
            print('Not Found match  %s! Distance: %1.4f' % (aligned_image_paths[i], dist))
