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
                click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
            else:
                # print(file)
                known_names.append(basename)
                known_face_encodings.append(encodings[0])
    return known_names, known_face_encodings


def get_result(filename, name, distance, show_distance=False):
    if show_distance:
        # print("{},{}".format(name, distance))
        return name
    else:
        # print("{},{}".format(name, distance))
        return name
def print_result(filename, name, distance, show_distance=False):
    if show_distance:
        print("{},{},{}".format(filename, name, distance))
    else:
        print("{}".format(name))


def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.1, show_distance=False):
    # unknown_image = face_recognition.load_image_file(image_to_check)
    unknown_image = image_to_check

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)
    d_name = ''

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)
        if True in result:
            # [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
            d_name = [get_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            d_name = get_result(image_to_check, "unknown_person", None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        d_name = get_result(image_to_check, "no_persons_found", None, show_distance)
    return(d_name[0])

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]



# @click.command()
# @click.argument('known_people_folder')
# @click.argument('image_to_check')
# @click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
# @click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
# @click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def call_me(name, weights, image_to_check, tolerance, show_distance):
    name_is = test_image(image_to_check, name, weights, tolerance, show_distance)
    return(name_is)
