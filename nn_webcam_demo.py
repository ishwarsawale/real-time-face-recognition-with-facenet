from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import facenet
import detect_face
import os
from os.path import join as pjoin
import copy
import math
from sklearn.svm import SVC
from sklearn.externals import joblib
import pymongo
from pymongo import MongoClient
import datetime
from tensorflow.python.platform import gfile
import numpy as np
import sys
import detect_and_align
import id_data
from scipy import misc
import re
import cv2
import argparse
import time
import twoFace
import pickle
from PIL import Image
import idlib
import dlib_embed
import urllib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def recom():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client.retail_db
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            HumanNames = os.listdir("./input_dir")
            HumanNames.sort()
            print('Loading feature extraction model')
            modeldir = './pre_model/20170511-185253.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename = './my_class/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model_svm, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            # print('load classifier file-> %s' % classifier_filename_exp)
            # model = load_model('my_model_with_unknown.h5')
            # model = load_model('my_model.h5')
            with open ('weights', 'rb') as fp:
                id_dataset = pickle.load(fp)
            print('load facenet weights')
            with open ('dlib_weights_names', 'rb') as fp:
                dlib_weight_names = pickle.load(fp)
            print('load dlib weights')
            with open ('dlib_weights', 'rb') as fp:
                dlib_weights = pickle.load(fp)
            print('load dlib knn model')
            video_capture = cv2.VideoCapture(0)
            counter = 1
            show_landmarks = True
            show_bb = True
            show_id = True
            show_fps = False
            print('Start Recognition!')
            prevTime = 0
            while True:
                start = time.time()
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(frame, pnet, rnet, onet)
                cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(frame)
                frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # print(bounding_boxes)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        try:
                            cropped[i] = facenet.flip(cropped[i], False)
                        except Exception as e:
                            continue
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        matching_id, dist = twoFace.find_matching_id(id_dataset, emb_array[0, :])
                        
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        predictions_svm = model_svm.predict_proba(emb_array)
                        # print(predictions_svm)
                        best_class_indices_svm = np.argmax(predictions_svm, axis=1)
                        # print(best_class_indices_svm)
                        best_class_probabilities_svm = predictions_svm[np.arange(len(best_class_indices_svm)), best_class_indices_svm]
                        # print(best_class_probabilities_svm)
                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        result_names = ''
                        result_names_svm = ''
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices_svm[0]] == H_i:
                                result_names_svm = HumanNames[best_class_indices_svm[0]]
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                frame_to_check = np.array(pil_im)
                                predictions = idlib.predict(frame_to_check, model_path="trained_knn_model.clf")
                                emb_name = dlib_embed.call_me(dlib_weight_names,dlib_weights, frame_to_check,0.3, False)
                                knn_name = ''
                                for name, (top, right, bottom, left) in predictions:
                                    knn_name = name
                                conf = (2 - dist) * 50 
                                # print(conf)
                                one_or = ''
                                if matching_id == knn_name:
                                    one_or = matching_id
                                if emb_name == result_names:
                                    one_or = emb_name
                                dist_threshold = 0.3
                                if (knn_name == result_names == result_names_svm) and (best_class_probabilities >= 0.70) and conf > 30 :
                                    # print ('recognized user is:', result_names)
                                    print("from KNN Dlib: ", knn_name)
                                    print("from weights Dlib:", emb_name)
                                    print ('from Facent NN :', result_names)
                                    print ('from Facent SVM :', result_names_svm)
                                    print('from embedding facenet ', matching_id)
                                    print('from facenet embedding distance: ', dist)
                                    print('probability score NN Facenet: ', best_class_probabilities)
                                    print(' one or : ', one_or)

                                    print('final value after at frame :', result_names)
                                    # for name, (top, right, bottom, left) in predictions:
                                        # print("- Found {} at ({}, {})".format(name, left, top))
                                    # print('probability score: ', best_class_probabilities)
                                    if show_id:
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(frame, result_names, (text_x, text_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                                else:
                                    print("from KNN Dlib: ", knn_name)
                                    print("from weights dlib_weights:", emb_name)
                                    print ('from Facent NN :', result_names)
                                    print ('from Facent SVM :', result_names_svm)
                                    print('from embedding facenet ', matching_id)
                                    print('from facenet embedding distance: ', dist)
                                    print('probability score NN Facenet: ', best_class_probabilities)
                                    print(' one or : ', one_or)                                    
                                    print('final value after at frame :', "unknown")
                                    if show_id:
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(frame, "unknown", (text_x, text_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                        if show_bb:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 0, 0), 2)    #boxing face

                        if show_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmarks[i, j]) - size, int(landmarks[i, j + 5]) - size)
                                bottom_right = (int(landmarks[i, j]) + size, int(landmarks[i, j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                    else:
                        print('\n')

                end = time.time()
                seconds = end - start
                fps = round(1 / seconds, 2)
                # if show_id:
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(frame, result_names, (text_x, text_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if show_fps:
                    text_fps_x = len(frame[0]) - 150
                    text_fps_y = 20
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(fps), (text_fps_x, text_fps_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    show_landmarks = not show_landmarks
                elif key == ord('b'):
                    show_bb = not show_bb
                elif key == ord('i'):
                    show_id = not show_id
                elif key == ord('f'):
                    show_fps = not show_fps
            video_capture.release()
            cv2.destroyAllWindows()


# recom()