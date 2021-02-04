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


def plot_accs_vs_gens(accs_dict, args):
    total_gen = args.n_cells * args.gen_per_cell
    gens_dict = {}
    for gen_index in range(total_gen):
        gens_dict[gen_index] = []
        for name, acc in accs_dict.items():
            if 'gen_{}_'.format(gen_index) in name:
                model_name = name.split('_')[-1]
                gens_dict[gen_index].append(acc)
    # Get average over generations
    gens_avg = {}
    for gen_index in range(total_gen):
        gens_avg[gen_index] = np.mean(gens_dict[gen_index])
    # Get max over generations
    gens_max = {}
    for gen_index in range(total_gen):
        gens_max[gen_index] = np.max(gens_dict[gen_index])
    # Plot avg and max vs generations
    gens = list(gens_avg.keys())
    gens = [g+1 for g in gens]
    avgs = list(gens_avg.values())
    plt.plot(gens, avgs, color='green', marker='o', linestyle='dashed', label='Average')
    maxs = list(gens_max.values())
    plt.plot(gens, maxs, color='blue', marker='x', linestyle='dashed', label='Best')
    plt.legend(loc='upper left')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    axes = plt.gca()
    axes.set_ylim([0.85,1])
    plt.show()

def train_model_from_descriptor(descriptor, model_name, train_gen, test_gen, args):
    # Build model
    model = ModelBuilder(cells_settings=descriptor,
                               filters_list=args.filters_list,
                               strides_list=args.strides_list,
                               settings=args,
                               n_blocks=args.n_blocks_per_cell,
                               n_blocks_per_block=args.n_subblocks_per_block).get_model()
    # Train the model
    # Define optimizer, metrics and loss first
    optimizer = keras.optimizers.SGD(lr=args.lr_start,
                                     momentum=args.momentum,
                                     decay=0.0,
                                     nesterov=False)
    metrics = ['accuracy']
    loss = keras.losses.CategoricalCrossentropy()
    # Compile model using optimizer, metrics and loss
    model.compile(optimizer=optimizer,
                      metrics=metrics,
                      loss=loss)
    # Define path where to save best model and add to callbacks
    gen_folder_trained_models = os.path.join(args.log_folder,
                                                  args.models_folder,
                                                  'final')
    if not os.path.exists(gen_folder_trained_models):
        os.makedirs(gen_folder_trained_models)
    model_path = os.path.join(gen_folder_trained_models,
                              model_name + '.h5')
    callbacks = get_standard_callbacks(args, model_path)
    # Fit model and get the final testing accuracy
    print('Fitting model, this may take a while...')
    model.fit(train_gen,
              validation_data=test_gen,
              epochs=args.n_epochs,
              callbacks=callbacks,
              verbose=0)
    # Get the trained best model from the h5 file
    trained_model = keras.models.load_model(model_path)
    return trained_model

def evaluate_model(model, model_name, test_gen):
    _, acc = model.evaluate(test_gen, verbose=0)
    print('Model {} -> Acc: {}'.format(model_name, acc))


if __name__ == '__main__':
    args = settings_parser.arg_parse()

    if False:
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

        plot_accs_vs_gens(accs_dict, args)

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
    model_name = 'found_best'
    model = train_model_from_descriptor(original_model_descriptor,
                                        model_name,
                                        train_gen,
                                        test_gen,
                                        args)
    evaluate_model(model, model_name, test_gen)
    first_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_0_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_2_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_2_block_1'], 'ops': ['3xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_3_block_0'], 'ops': ['1xconv']},
                                          {'ID': '2', 'in': ['cell_3_block_1'], 'ops': ['3xconv']}]}]
    model_name = '1st_cell_x4'
    model = train_model_from_descriptor(first_model_descriptor,
                                        model_name,
                                        train_gen,
                                        test_gen,
                                        args)
    evaluate_model(model, model_name, test_gen)
    second_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['5xconv']},
                                           {'ID': '1', 'in': ['cell_0_block_0'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['5xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['5xconv']},
                                           {'ID': '1', 'in': ['cell_1_block_0'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['5xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['5xconv']},
                                           {'ID': '1', 'in': ['cell_2_block_0'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_2_block_1'], 'ops': ['5xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['5xconv']},
                                           {'ID': '1', 'in': ['cell_3_block_0'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_3_block_1'], 'ops': ['5xconv']}]}]
    model_name = '2nd_cell_x4'
    model = train_model_from_descriptor(second_model_descriptor,
                                         model_name,
                                         train_gen,
                                         test_gen,
                                         args)
    evaluate_model(model, model_name, test_gen)
    third_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['model_input'], 'ops': ['5xconv']},
                                          {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['1xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_0_out'], 'ops': ['5xconv']},
                                          {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['1xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_1_out'], 'ops': ['5xconv']},
                                          {'ID': '2', 'in': ['cell_2_block_1'], 'ops': ['1xconv']}]},
                              {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['3xconv']},
                                          {'ID': '1', 'in': ['cell_2_out'], 'ops': ['5xconv']},
                                          {'ID': '2', 'in': ['cell_3_block_1'], 'ops': ['1xconv']}]}]
    model_name = '3rd_cell_x4'
    model = train_model_from_descriptor(third_model_descriptor,
                                        model_name,
                                        train_gen,
                                        test_gen,
                                        args)
    evaluate_model(model, model_name, test_gen)
    fourth_model_descriptor = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['1xconv']},
                                           {'ID': '1', 'in': ['model_input'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_0_block_1'], 'ops': ['1xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['1xconv']},
                                           {'ID': '1', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_1_block_1'], 'ops': ['1xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['1xconv']},
                                           {'ID': '1', 'in': ['cell_1_out'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_2_block_1'], 'ops': ['1xconv']}]},
                               {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['1xconv']},
                                           {'ID': '1', 'in': ['cell_2_out'], 'ops': ['3xconv']},
                                           {'ID': '2', 'in': ['cell_3_block_1'], 'ops': ['1xconv']}]}]
    model_name = '4th_cell_x4'
    model = train_model_from_descriptor(fourth_model_descriptor,
                                         model_name,
                                         train_gen,
                                         test_gen,
                                         args)
    evaluate_model(model, model_name, test_gen)
