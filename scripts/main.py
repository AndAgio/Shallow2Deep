import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import copy
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from keras import regularizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from tensorflow.python.keras.engine import data_adapter
# Imported from files
import settings_parser
from utils import *
from dataset_importer import *
from sota_models import *
from train_helper import *
from model_footprint import *
from model_builder import ModelBuilder
from evolution import Population


if __name__ == '__main__':
    args = settings_parser.arg_parse()

    # Setup the file containing settings first
    if args.start.lower() in ('yes', 'true', 't', 'y', '1'):
        if args.dataset in ['mnist', 'fashion']:
            args.img_channels = 1
            args.img_shape = 28
        elif args.dataset in ['cifar']:
            args.img_channels = 3
            args.img_shape = 32
        my_population = Population(args)
        file_path = os.path.join(args.log_folder, 'continuation_settings')
        file = open(file_path, "wb")
        pickle.dump(args, file)
        file.close()
    elif args.start.lower() in ('no', 'false', 'f', 'n', '0'):
        file_path = os.path.join(args.log_folder, 'continuation_settings')
        file = open(file_path, "rb")
        args = pickle.load(file)
        file.close()
        # Restart evolution from history
        prev_history_path = os.path.join(args.log_folder, 'history_log_pickle')
        print('Log folder: {}'.format(args.log_folder))
        my_population = Population(args, restart=True, prev_h=prev_history_path)
    else:
        raise ValueError('The settings for start is not a valid setting.'
                         'It can either be True/False, Yes/No or 1/0')

    #out_file = os.path.join(args.log_folder, 'models weights.txt')
    #print_all_models_weights(args.log_folder, args.models_folder, out_file)
    # Import data
    if args.dataset == 'cifar':
        data = get_generator_from_cifar(args, split_train=True, small=False)
    elif args.dataset == 'mnist':
        data = get_generator_from_mnist(args, split_train=True, small=False)
    elif args.dataset == 'fashion':
        data = get_generator_from_fashion_mnist(args, split_train=True, small=False)
    else:
        raise ValueError('Not a valid dataset!')

    # Run evolution
    #my_population.run_evolution(data)
    # Run batched evolution
    my_population.run_batched_evolution(data)
