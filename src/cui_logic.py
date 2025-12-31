"""Command Driven Logic for sim"""

import sys
import os
# import automotive_ice_modelling_v2 as sim
import utils


# Create paths
root_path = os.getcwd() + "/"

def opt1():
    print(1)

test = {'option 1': opt1,
        'option 2': print}

print(test)

def option(window):
    print(window.keys())
    command = input("> ")
    try:
        window[command]()
    except KeyError:
        print("invalid command, please try again")
        option(window)
    return command

option(test)

# functions
def new_model():
    """generat enew model"""
    print("new model function")
    model_name = input("New model name")
    os.mkdir('models/' + model_name)
    model_path = 'models/' + model_name + '/'
    os.mkdir(model_path + 'inputs')
    os.mkdir(model_path + 'outputs')
    print("new model directory created")
    os.mkdir(model_path + 'outputs/graphs')


def open_model():
    model_dict = {}
    print("open model function")
    models_path = root_path + "models/"
    list_models = os.listdir(models_path)
    for model in list_models:
        print(f"{model}")
        exact_path = models_path + model
        model_files = os.listdir(exact_path)
        model_dict[model] = model_files
    print("models found:")
    cui_move(model_dict)


def quit_option():
    yes_list = {'yes', 'y', 'ye', ''}
    no_list = {'no', 'n'}
    run = True
    while run is True:
        ans = input("do you wish to end the program?")
        if ans.lower() in yes_list:
            run = False
        elif ans.lower in no_list:
            run = True
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")


# windows
# intro_menu = {'new model': new_model,
#               'open model': open_model,
#               'end': sys.exit()}

intro_menu = {'new model': print("option 1"),
              'open model': print("option 2"),
              'end': sys.exit()}


# main menu to call functions
main_menu = {'run sim': sim.main,
             'create graphs': utils.create_graphs,
             'read output': sim.read_output,
             'end': sys.exit()}


def cui_move(window):
    print(window.keys())
    command = input("> ")
    try:
        window[command]()
    except KeyError:
        print("invalid command, please try again")
        option(window)
    return command


# print(intro_menu.keys())
# option(intro_menu)
option(test)