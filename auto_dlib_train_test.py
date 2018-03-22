import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import re
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def predict(train_dir, mode, knn_clf=None, model_path=None, distance_threshold=0.6, verbose=False):
    """
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=102)
    # Load a trained KNN model (if one was passed in)
    if(mode == 'TRAIN'):
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree', weights='distance')
        knn_clf.fit(X_train, y_train)
        if model_path is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(knn_clf, f)
    if(mode == 'TEST'):
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
        acc = knn_clf.score(X_test, y_test)
        print("accuracy: {:.2f}%".format(acc * 100))



if __name__ == '__main__':
    # predict('input_dir_small',mode='TRAIN', model_path="./my_class/trained_knn_model_small.clf")
    predict('input_dir', mode='TEST', model_path="./my_class/trained_knn_model_small.clf")