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

menu_actions  = {}
# Main menu
def main_menu():
    os.system('clear')

    print ("Welcome,\n")
    print ("Please choose the menu you want to start:")
    print ("1. Face Recognition Using WebCam")
    print ("2. Face Recognition Using Android Mobile")
    print ("3. Create Training Data for User")
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
    print ("2. Face Recognition Using Android Mobile")
    print ("3. Create Training Data for User")
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

# Menu 2
def menu3():
    print('\n')
    print('-----------------------------------------------------------')
    print('opening http://0.0.0.0:5000 to add data for new user')
    print('-----------------------------------------------------------')
    print('\n')
    url = 'http://0.0.0.0:5000'
    if sys.platform == 'darwin':    # in case of OS X
        subprocess.Popen(['open', url])
    else:
        webbrowser.open_new_tab(url)
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
    '9': back,
    '0': exit,
}
if __name__ == "__main__":
    main_menu()
    