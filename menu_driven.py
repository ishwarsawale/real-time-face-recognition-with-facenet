from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import nn_webcam_demo as demo
import android_stream_demo as an_demo
import webbrowser
import subprocess
import aligndata_first as align
import create_dlib_weight as dweight
import create_weights as fweight
import train_nn_face as tface
import create_classifier_se as tsvm
import train_dlib_knn as tknn

menu_actions  = {}
# Main menu
def main_menu():
    os.system('clear')

    print ("Welcome,\n")
    print ("Please choose the menu you want to start:")
    print ("1. Face Recognition Using WebCam")
    print ("2. Face Recognition Using Android Cam")
    print ("3. Create Training Data for Users")
    print ("4. Train All Models for Users")
    print ("\n0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

    return

def custom_menu(msg):
    os.system('clear')
    print('--------------------------------------')
    print(msg)
    print('--------------------------------------')
    print ("Please choose the menu you want to start:")
    print ("1. Face Recognition Using WebCam")
    print ("2. Face Recognition Using Android Cam")
    print ("3. Create Training Data for Users")
    print ("4. Train All Models for Users")
    print ("\n0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

    return

# Execute menu
def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print ("Invalid selection, please try again.\n")
            menu_actions['main_menu']()
    return

# Menu 1
def menu1():
    print ("Start Face Recognition !\n")
    demo.recom()
    print ("9. Back")
    print ("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return


# Menu 2
def menu2():
    an_demo.recom()
    print ("9. Back")
    print ("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return

# Menu 3
def menu3():
    print('\n')
    print('-----------------------------------------------------------')
    print('opening https://0.0.0.0:5000 to add data for new user')
    print('-----------------------------------------------------------')
    print('\n')
    url = 'https://0.0.0.0:5000'
    if sys.platform == 'darwin':    # in case of OS X
        subprocess.Popen(['open', url])
    else:
        webbrowser.open_new_tab(url)
    print ("9. Back")
    print ("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return

# Menu 4
def menu4():
    align.align_data()
    dweight.dlib_weights()
    fweight.facenet_128D()
    tface.train_nn()
    tsvm.train_svm()
    tknn.train_knn()
    print ("9. Back")
    print ("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return

def back():
    menu_actions['main_menu']()

def exit():
    sys.exit()

menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '3': menu3,
    '4': menu4,
    '9': back,
    '0': exit,
}
if __name__ == "__main__":
    main_menu()
    