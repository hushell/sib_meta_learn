import os
import json
import yaml
import argparse
from easydict import EasyDict

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict

def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        config_dict = yaml.load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config, config_dict

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-k', '--steps',
        default=3,
        type=int,
        help='The number of SIB steps')
    argparser.add_argument(
        '-s', '--seed',
        default=100,
        type=int,
        help='The random seed')
    argparser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='GPU id')
    args = argparser.parse_args()
    return args

def get_config():
    args = get_args()
    config_file = args.config

    if config_file.endswith('json'):
        config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    if not hasattr(config, 'seed'): config.seed = args.seed
    config.gpu = args.gpu
    config.nStep = args.steps
    config.cacheDir = os.path.join("cache", '{}_{}shot_K{}_seed{}'.format(
        config.expName, config.nSupport, config.nStep, config.seed))
    config.logDir = os.path.join(config.cacheDir, 'logs')
    config.outDir = os.path.join(config.cacheDir, 'outputs')

    # create the experiments dirs
    create_dirs([config.cacheDir, config.logDir, config.outDir])

    return config
