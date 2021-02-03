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

    # Get file containing history log
    file_path = os.path.join(args.log_folder, 'history_log_pickle')
    file = open(file_path, "rb")
    history = pickle.load(file)
    file.close()

    #Restore settings
    file_path = os.path.join(args.log_folder, 'continuation_settings')
    file = open(file_path, "rb")
    args = pickle.load(file)
    file.close()

    # Create dictionary containing the name and accuracies of models
    accs_dict = {}
    for name, subdict in history.items():
        accs_dict[name] = subdict['Accuracy']

    # Get name of model with highest accuracy and its structure
    best_name = max(accs_dict, key=accs_dict.get)
    best_model_descriptor = history[best_name]['Descriptor']

    # Build model
    best_model = ModelBuilder(cells_settings=best_model_descriptor,
                               filters_list=args.filters_list,
                               strides_list=args.strides_list,
                               settings=args,
                               n_blocks=args.n_blocks_per_cell,
                               n_blocks_per_block=args.n_subblocks_per_block).get_model()

    # Import data
    if args.dataset == 'cifar':
        train_gen, test_gen = get_generator_from_cifar(args, split_train=True, small=False)
    elif args.dataset == 'mnist':
        train_gen, test_gen = get_generator_from_mnist(args, split_train=True, small=False)
    elif args.dataset == 'fashion':
        train_gen, test_gen = get_generator_from_fashion_mnist(args, split_train=True, small=False)
    else:
        raise ValueError('Not a valid dataset!')

    # Overwrite training settings
    args.n_epochs = 60
    args.lr_start = 0.01
    args.lr_decay_epochs = 20
    args.lr_decay_factor = 10

    # Train the model
    # Define optimizer, metrics and loss first
    optimizer = keras.optimizers.SGD(lr=args.lr_start,
                                     momentum=args.momentum,
                                     decay=0.0,
                                     nesterov=False)
    metrics = ['accuracy']
    loss = keras.losses.CategoricalCrossentropy()
    # Compile model using optimizer, metrics and loss
    best_model.compile(optimizer=optimizer,
                  metrics=metrics,
                  loss=loss)
    # Define path where to save best model and add to callbacks
    gen_folder_trained_models = os.path.join(args.log_folder,
                                                  args.models_folder,
                                                  'final_best')
    if not os.path.exists(gen_folder_trained_models):
        os.makedirs(gen_folder_trained_models)
    model_path = os.path.join(gen_folder_trained_models,
                              'found_best_trained.h5')
    callbacks = get_standard_callbacks(args, model_path)
    # Fit model and get the final testing accuracy
    best_model.fit(train_gen,
              validation_data=test_gen,
              epochs=args.n_epochs,
              callbacks=callbacks,
              verbose=1)

    # Train model with only first replicated 4 times
    original_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']},
                                             {'ID': '1', 'in': ['cell_0_block_0'], 'ops': ['1xconv']},
                                             {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['3xconv']}]},
                                 {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['5xconv']},
                                             {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['3xconv']},
                                             {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['5xconv']}]},
                                 {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xconv']},
                                             {'ID': '1', 'in': ['cell_1_out'], 'ops': ['5xconv']},
                                             {'ID': '2', 'in': ['cell_2_block_1'], 'ops': ['1xconv']}]},
                                 {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['1xconv']},
                                             {'ID': '1', 'in': ['cell_2_out'], 'ops': ['3xconv']},
                                             {'ID': '2', 'in': ['cell_3_block_1'], 'ops': ['1xconv']}]}]
    first_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_0_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['3xconv']}]}]
