import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob
import json
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import from files
import settings_parser


def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=100, fill='=', print_end="\r"):
    """
    Call in a loop to create terminal progress bar.
    Params:
        - iteration: current iteration (Int)
        - total: total iterations (Int)
        - prefix: prefix string (Str)
        - suffix: suffix string (Str)
        - decimals: positive number of decimals in percent complete (Int)
        - length: character length of bar (Int)
        - fill: bar fill character (Str)
        - print_end: end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar_bar = fill * filled_length + ' ' * (length - filled_length)
    print(f'\r{prefix} [{bar_bar}] {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


class SpecialStyles():
    """
    Class that defines sequences of styles/colors that can be used in prints.
    """

    def __init__(self):
        self.purple = '\033[95m'
        self.cyan = '\033[96m'
        self.darkcyan = '\033[36m'
        self.blue = '\033[94m'
        self.green = '\033[92m'
        self.yellow = '\033[93m'
        self.red = '\033[91m'
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.end = '\033[0m'


def print_bold(message_string):
    """
    Function to print a message in bold characters.
    """
    print(SpecialStyles().bold + message_string + SpecialStyles().end)


def print_warning(message_string):
    """
    Function to print a message in yellow characters as a warning.
    """
    print(SpecialStyles().yellow + message_string + SpecialStyles().end)


def blockPrint():
    '''
    Function used to disable print on console
    '''
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    '''
    Function used to enable print on console
    '''
    sys.stdout = sys.__stdout__


def get_out_shape(input_shape, operation, n_filters, stride):
    """
    Function to get the output shape of an operation applied to a certain input shape.
    Params:
        - input_shape: Tuple specifying the shape of the input.
        - operation: String specifying the operation to be applied.
        - n_filters: Number of filters that this layer wants to use.
        - stride: Integer identifying the stride value used for convolutional layers.
    Returns:
        - Tuple specifying the shape of the output
    """
    if operation in ['identity', '3xsqueezeexicte']:
        output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    elif operation in ['3xdepthwise', '5xdepthwise', '1xconv', '3xconv', '5xconv', '3xdilated',
                       '3xmax', '3xaverage', '3xshuffle', '3xinvmobile', '5xinvmobile']:
        if stride == 1:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], n_filters)
        elif stride == 2:
            output_shape = (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, n_filters)
        else:
            raise ValueError('Stride value is not 1 nor 2 checks again the implementation of this block!')
    else:
        raise ValueError('Operation you select doesn\'t belong to the search space')
    return output_shape


def check_inputs_shapes(inputs):
    """
    Function that checks if all inputs have the same shape.
    Params:
        - inputs: Dictionary of tensors.
    Returns:
        - Boolean, True if all tensors have the same shape, False otherwise.
    """
    valid = True
    shapes = []
    for _, input in inputs.items():
        shapes.append(tuple(input.get_shape().as_list()))
    #print('Shapes before set: {}'.format(shapes))
    shapes = list(set(shapes))
    #print('Shapes after set: {}'.format(shapes))
    if len(shapes) != 1:
        valid = False
    return valid


def get_unique_input_shape(inputs):
    """
    Function that returns the input shape if they are all the same,
    otherwise it returns None.
    Params:
        - inputs: Dictionary of tensors.
    Returns:
        - Int or None.
    """
    shapes = []
    for _, input in inputs.items():
        shapes.append(tuple(input.get_shape().as_list()))
    print('Shapes before set: {}'.format(shapes))
    shapes = list(set(shapes))
    print('Shapes after set: {}'.format(shapes))
    if len(shapes) != 1:
        return None
    return shapes[0]


def get_inputs_shape_dictionary(inputs):
    """
    Function that returns a dictionary containing the name of the output layer
    and the corresponding shape. It returns also the name of the input layer having
    the smallest input height x width.
    Params:
        - inputs: Dictionary of tensors.
    Returns:
        - Dictionary of tuples (shapes).
    """
    shapes = {}
    smallest = 100000
    smallest_shape = None
    for key, input in inputs.items():
        input_h = input.get_shape().as_list()[1]
        if input_h < smallest:
            smallest_shape = tuple(input.get_shape().as_list())
        shapes[key] = tuple(input.get_shape().as_list())
    print('Shapes dictionary: {}'.format(shapes))

    return shapes, smallest_shape


def check_substring_in_list(substring, list):
    '''
    Function used to check if a substring is present in a list of strings
    Params:
        - substring: String that needs to be serched inside each string of the list.
        - list: list of strings used as the search space.
    Returns:
        - Boolean: True if the substring exists in a string of the list
    '''
    for string in list:
        if substring in string:
            return True
    return False


def get_strides_from_input_strings(substring, list):
    '''
    Function used to convert a list of strings into the corresponding list of strides.
    Params:
        - substring: string to search in each string of the list. Here the substring
                     to search is 'block', because this indicates that this block
                     receives inputs from a previous block in the cell, therefore
                     we want to use stride 1 in those block to allow operations
                     to be completed.
        -list: List of strings to be used as search space.
    Returns:
        - List of 1s and 2s indicating the strides that we want to use for each block.
    '''
    strides_list = []
    for string in list:
        if substring in string:
            strides_list.append(1)
        else:
            strides_list.append(2)
    return strides_list


def remove_later_blocks_from_inputs(possible_inputs, id):
    '''
    Function used to remove from the possible inputs
    dictionary all blocks of this cell with id greater
    than the one of the block that we are considering now.
    Params:
        - possible_inputs: Dictionary containing all inputs
                           value for the current block to mutate.
        - id: String identifying the block that we want to mutate.
    Returns:
        - Dictionary with new possible inputs.
    '''
    # Define an empty dictionary
    skimmed_inputs = {}
    # Iterate over all possible inputs
    for key, value in possible_inputs.items():
        #print('Input: {}'.format(key))
        # If input comes from previous cells is ok to keep it
        if 'block' not in key:
            skimmed_inputs[key] = value
        # Otherwise keep only output of earlier blocks
        else:
            block_id = int(key.split('block_')[-1])
            id = int(id)
            if block_id < id:
                skimmed_inputs[key] = value
    return skimmed_inputs


def plot_image_from_generator(generator, name):
    '''
    Function that plots and saves to a certain file the first image
    of a generator that is used for training, validating or testing a model.
    Params:
        - generator: ImageDataGenerator object used to fit keras models.
        - name: name of the picture that we want to save. Path/name.png
    '''
    img = generator.next()[0][0]
    plt.imshow(img)
    plt.savefig(name)
