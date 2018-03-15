# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import pickle


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []
    folder_list = os.listdir(known_people_folder)

    for folder in folder_list:
        images = os.listdir(known_people_folder+'/'+folder)
        for file in images:
            basename = folder
            file = known_people_folder + '/' + folder + '/' + file
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 1:
                click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

            if len(encodings) == 0:
                os.remove(file)
                click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
                print('removed file %s' % file)
            else:
                print(file)
                known_names.append(basename)
                known_face_encodings.append(encodings[0])
    return known_names, known_face_encodings


def dlib_weights():
    names, weights = scan_known_people('./input_dir')
    with open('dlib_weights_names', 'wb') as fp:
        pickle.dump(names, fp)
    with open('dlib_weights', 'wb') as fp:
        pickle.dump(weights, fp)

dlib_weights()