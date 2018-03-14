#!/bin/bash
while [ 1 -eq 1 ]  # or while [ true ] ( as glenn said)
do
 select opt in "${options[@]}"
 do
 case $opt in
     "Align Images")
         echo "You chose Create Group 1"
         ;;
     "Train facenet")
         echo "You chose to Delete Group"
         ;;
     "Train Dlib")
         echo "You chose to Create User"
         ;;
     "Train all")
        echo "you chose to train all models"
        # echo "align images first"
        # python aligndata_first.py
        echo "training first nn and facenet based model"
        python train_nn_face.py
        echo "training knn and dlib based model"
        python train_dlib_knn.py
        echo "training svm and facenet based model"
        python create_classifier_se.py
        echo "creating facenet embedding for users"
        python create_weights.py
        echo "creating dlib_weights"
        python create_dlib_weight.py
         ;;
     "Quit")
         break
         ;;
     *) echo invalid option;;
  esac
 done
done
# PS3='Please enter your choice: '
# options=("Option 1" "Option 2" "Option 3" "Quit")
# select opt in "${options[@]}"
# do
#     case $opt in
#         "Option 1")
#             echo "you chose choice 1"
#             ;;
#         "Option 2")
#             echo "you chose choice 2"
#             ;;
#         "Option 3")
#             echo "you chose to train all models"
#             echo "align images first"
#             python aligndata_first.py
#             echo "training first nn and facenet based model"
#             python train_nn_face.py
#             echo "training knn and dlib based model"
#             python train_dlib_knn.py
#             echo "training svm and facenet based model"
#             python create_classifier_se.py
#             echo "creating facenet embedding for users"
#             python create_weights.py
#             echo "creating dlib_weights"
#             python create_dlib_weight.py
#             ;;
#         "Quit")
#             break
#             ;;
#         *) echo invalid option;;
#     esac
# done