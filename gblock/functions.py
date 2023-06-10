import yaml
import logging
import os
from datetime import datetime

def print_function(input_string, **kwargs):
    print(input_string + '...')
    
    for key, arg in kwargs.items():
        len_key = len(key)
        len_arg = len(str(arg))
        n_dots = 50 - len_key - len_arg
        print(2*'-', key, n_dots*'.', arg)

    print(50*'_')

def print_to_log(input_string, path = None, file_name = None):

    if path is None:
        path = os.getcwd()

    if file_name is None:
        file_name = 'log.txt'
    
    dir = path + '/' + file_name

    date_time_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(dir, 'a') as f:
        f.write(str(date_time_string) + ' | ' + str(input_string) + '\n')

def load_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict


def initiate_logger(logname):

    return logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s | %(msecs)d | %(name)s | %(levelname)s | %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        level=logging.DEBUG)