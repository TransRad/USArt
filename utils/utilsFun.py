import json
from bunch import Bunch
import os
import tensorflow as tf


# Loss function
def ssim(y_true, y_pred):
    """
    ssim loss function
    :param y_true: ground truth image
    :param y_pred: predicted image
    :return: ssim parameter
    """
    # max_pixel = tf.math.reduce_max(y_true)
    # min_pixel = tf.math.reduce_min(y_true)
    y_true_new = y_true * 255.0 / tf.keras.backend.max(y_true)
    y_pred_new = y_pred * 255.0 / tf.keras.backend.max(y_pred)
    return tf.reduce_mean(tf.image.ssim(y_true_new, y_pred_new, 2.0))


def get_int(name, config):
    """
    return index of the name from the par object in JSON file
    :param name:
    :param config:
    :return: int of the name
    """
    ind = get_index(name, config.name_list)
    return config['par'][ind]['int']


def get_index(name, list_names):
    """
    return index of the name from the par object in JSON file
    :param list_names:
    :param name:
    :return: index
    """
    return list_names.index(name)


def open_json(json_file):
    """
    Open json file
    :param json_file:
    :return: config_json
    """
    config_json = []
    try:
        config_json = process_config(json_file)
        # print("\t json path fine")
    except:
        print("missing or invalid arguments")
    return config_json


def process_config(json_file):
    """
    Process the json file
    :param json_file:
    :return: config
    """
    config = get_config_from_json(json_file)
    config.name_list = get_par_names(config)
    config.tun_id = 0

    path = config.output_dir_base
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)

    return config


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(name) or config(dictionary)
    """
    # print("get parameters from the JSON file")
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return Bunch(config_dict)


def get_par_names(config):
    """
    Get all names for the par object in JSON file
    :param config:
    :return: names list
    """
    sz = len(config.par)
    return [config['par'][i]['name'] for i in range(0, sz)]



