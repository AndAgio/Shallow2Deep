import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import ast
import math
import random
import copy
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import regularizers
from keras.utils import plot_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, SeparableConv2D, ReLU
from keras.layers import MaxPooling2D, Activation, BatchNormalization, Concatenate
from keras.layers import AveragePooling2D, DepthwiseConv2D, GlobalAveragePooling2D
# Import from files
import settings_parser
from utils import *
from train_helper import *
from model_footprint import get_memory_footprint


class SearchSpace():
    """Class used to the define the search space."""
    def __init__(self, file):
        '''
        Initialization method used to build a dictionary that contains
        the filters list to be searched during evolution.
        Params:
            - file: Path to a file containing the dictionary for the search space.
        '''
        self.space = {}
        self.initialize(file)

    def add_key(self, key):
        '''
        Method used to add a key to the search space dictionary.
        Params:
            - key: String indicating the key that needs to be added to the dictionary.
        '''
        # If the passed key is already in the dictionary raise an error
        if key in self.space.keys():
            raise ValueError('Passed key is already in the search space')
        self.space[key] = []

    def add_parameter(self, key, parameter):
        '''
        Method used to add parameters to the search space dictionary, through a key.
        Params:
            - key: String indicating the key to expand or to add to dictionary.
            - parameter: List or single value to add to the dictionary.
        '''
        # If the key and the parameter is already in the search space raise an error
        if key in self.space.keys() and parameter in self.space[key]:
            raise ValueError('Passed key and parameters is already in the search space')
        # If the parameter is a list add single values individually
        if key in self.space.keys() and type(parameter) == list:
            for par in parameter:
                self.space[key].append(par)
        # If the parameter is a single value append it
        elif key in self.space.keys():
            self.space[key].append(parameter)
        # If the key is not already in the dictionary create the new key's list
        if key not in self.space.keys():
            self.space[key] = []
            self.space[key] = parameter

    def initialize(self, file_path):
        '''
        Method used to read the search space file and build the
        corresponding search space dictionary.
        Params:
            - file_path: Path to the text file containing the search space.
        '''
        file = open(file_path, "r")
        contents = file.read()
        # Convert content to dictionary
        self.space = ast.literal_eval(contents)
        # print('Search space: {}'.format(self.space))

    def get_random_value(self, key):
        '''
        Method to extract a random value from the dictionary of the
        search space given a certain key.
        Params:
            - key: String identifying the key to seasrch from.
        Returns:
            - String or Integer corresponding to the random chosen value.
        '''
        if key not in self.space.keys():
            raise ValueError('Passed key doesn\'t exists in the search space')
        return random.choice(self.space[key])

    def check_kernel(self, ker):
        '''
        Mathod to check if a certain kernel is inside the list of
        kernels to be searched from.
        Params:
            - ker: String identifying the be looking for in the search space.
        Returns:
            - Boolean, True if the kernel is inside search space, False otherwise.
        '''
        if ker in self.space['kernels']:
            return True
        return False

    def get_kernels(self):
        '''
        Method used to get all kernels available inside the search space.
        Returns:
            - List of all strings in the kernels list.
        '''
        return self.space['kernels']

    def copy(self):
        return copy.deepcopy(self)


class BlockStructure():
    """
    Class used to the define the structure of a single block. The block is
    composed of a single layer or multiple layers in parallel. If multiple layers
    are selected then their outputs are summed at the output of the block.
    Block's operations and inputs shape are checked in order to avoid any
    possible inconsistency in terms of input-output structure.
    """
    def __init__(self, full_block_name, search_space, inputs, input_strings,
                 operations, n_filters=64, strides_list=None):
        """
        Constructor of block. We need to pass the ID of the block that we're
        building, the search space, the shapes of the inputs that we want to use.
        If no n_components is selected then 2 layers are used as default.
        If no operations are passed then they are selected randomly from
        the search space.
        Params:
            - full_block_name: String that identifies uniquely a block of a cell.
                               It should be of the form: cell_0_block_1
            - search_space: Object containing the possible values for
                            the search space.
            - inputs: List of tensors containing the inputs of the block that
                      we want to build.
            - input_strings: List of strings containing the inputs of the
                             block as IDs of other blocks.
            - operations: List of operations to be done on the layers of this block.
                          A sanity check is made on the list to be sure they
                          belong to search space and they're compatible with
                          input_shapes, to avoid output shapes conflicts.
            - n_filters: Integer identifying the number of filters to use
                         in each convolutional layer of this block.
            - stride: Integer identifying the stride value used for
                      convolutional layers of this block.
        """
        # Define hyperparameters
        self.full_block_name = full_block_name
        self.search_space = search_space
        self.n_components = len(inputs)
        self.input_strings = input_strings
        self.inputs = inputs
        self.operations = operations
        self.n_filters = n_filters
        # Define alpha used for kernels reguralization
        self.alpha = 4e-5
        # Check if strides_list is already defined, otherwise set all to 1 by default
        if strides_list is not None:
            self.strides_list = strides_list
        else:
            self.strides_list = [1 for _ in range(self.n_components)]
        # Get input shapes and check if the operations are fine with input and output shapes
        input_shapes = [tens.shape for tens in inputs]
        self.valid = self.check_ops(input_shapes,
                                    self.operations,
                                    self.n_filters,
                                    self.strides_list)
        # If the block is valid build it, otherwise don't do anything
        if self.valid:
            self.build_block()
        else:
            pass

    def check_ops(self, input_shapes, operations, n_filters, strides_list):
        """
        Method that checks if the inputs and the corresponding operations are valid.
        Params:
            - inputs: List of integers that selects the positions of the
                      input in the cell
            - operations: List of strings identifing operations to be
                          done on each input.
            - n_filters: Integer identifying the number of filters to
                         use in each convolutional layer of this block.
            - stride: Integer identifying the stride value used for
                      convolutional layers of this block.
        Returns:
            - Boolean: True if operations and inputs are valid, False otherwise.
        """
        # First check if number of inputs and operations are the same
        if len(input_shapes) != len(operations):
            return False
        # Iterate over all subblocks...
        output_shapes = []
        for i in range(len(input_shapes)):
            input_shape = input_shapes[i]
            operation = operations[i]
            # ...and get the corresponding output shape
            output_shapes.append(get_out_shape(input_shape,
                                               operation,
                                               n_filters,
                                               strides_list[i]))
        # Check if all outputs have the same shape. If so the block is fine
        unique_output_shapes = list(set(output_shapes))
        if len(unique_output_shapes) == 1:
            return True
        else:
            return False

    def build_block(self):
        '''
        Method used to build the block from the settings passed to the constructor
        and checked for their consistency.
        '''
        # Define initial empty outputs
        outs = []
        # Iterate over each subblock
        for index in range(self.n_components):
            # Define the name of the subblock and get input and operation
            name = self.full_block_name + '_subblock_{}'.format(index)
            input = self.inputs[index]
            operation = self.operations[index]
            # Apply the operation and append the output to the list
            outs.append(self.apply_operation(input,
                                             operation,
                                             name,
                                             stride=self.strides_list[index]))
        # If only one output then this will be also the output of the block
        if len(outs) == 1:
            self.out = outs[0]
        # Otherwise all outputs are summed to get the ouput of the block
        else:
            self.out = tf.add_n(outs, name='out_block_{}'.format(self.full_block_name))

    def apply_operation(self, input_x, op_string, name, stride):
        '''
        Method used to call the different layers and apply them to the selected input.
        Params:
            - input_x: Tensor that is the input for the subblock that is being built.
            - op_string: String identifying the kernel to be applied to the input.
            - name: Name of the subblock being built.
            - stride: Stride value (1 if normal cell, 2 if reduction is needed).
        Returns:
            - Tensor output of the kernel applied to the input.
        '''
        #print('Applying {}...'.format(op_string))
        # Check if the operation is identity
        if op_string == 'identity':
            # Identity and stride are not available together. If stride is needed
            # with identity use a 1x1 convolutional filter with stride 2.
            if stride == 2:
                output_block = self.conv_layer(input_x,
                                               kernel_size=1,
                                               n_filters=self.n_filters,
                                               strides=stride,
                                               layer_name=name)
            else:
                output_block = input_x
        # Check if the operation is depthwise convolution
        elif op_string in ['3xdepthwise', '5xdepthwise']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            #print('Kernel size: {}'.format(k_size))
            # Apply depthwise convolution
            output_block = self.depthwise_conv_layer(input_x,
                                                     kernel_size=k_size,
                                                     n_filters=self.n_filters,
                                                     strides=stride,
                                                     layer_name=name)
        # Check if the operation is dilated convolution
        elif op_string in ['3xdilated']:
            # Apply dilated convolution. It is a simple convolution
            # with dilation rate set to 2
            output_block = self.dilation_conv_layer(input_x,
                                                    dilation=2,
                                                    strides=stride,
                                                    n_filters=self.n_filters,
                                                    layer_name=name)
        # Check if the operation is convolution followed by max pooling
        elif op_string in ['3xmax', '5xmax']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            # Apply the operation
            output_block = self.conv_max_layer(input_x,
                                               kernel_size=k_size,
                                               n_filters=self.n_filters,
                                               strides=stride,
                                               layer_name=name)
        # Check if the operation is convolution followed by average pooling
        elif op_string in ['3xaverage', '5xaverage']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            # Apply the operation
            output_block = self.conv_average_layer(input_x,
                                                   kernel_size=k_size,
                                                   n_filters=self.n_filters,
                                                   strides=stride,
                                                   layer_name=name)
        # Check if the operation is convolution followed by nothing
        elif op_string in ['1xconv','3xconv','5xconv']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            # Apply the operation
            output_block = self.conv_layer(input_x,
                                           kernel_size=k_size,
                                           n_filters=self.n_filters,
                                           strides=stride,
                                           layer_name=name)
        # Check if the operation is squeeze and excite convolution
        elif op_string in ['3xsqueezeexicte']:
            # Apply the operation
            output_block = self.squeeze_excitation_layer(input_x,
                                                         ratio=2,
                                                         n_filters=self.n_filters,
                                                         strides=stride,
                                                         layer_name=name)
        # Check if the operation is shuffled convolution
        elif op_string in ['3xshuffle', '5xshuffle']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            # Apply the operation
            output_block = self.shuffle_conv_layer(input_x,
                                                   kernel_size=k_size,
                                                   num_groups=2,
                                                   n_filters=self.n_filters,
                                                   strides=stride,
                                                   layer_name=name)
        # Check if the operation is inverted mobile bottleneck convolution
        elif op_string in ['3xinvmobile', '5xinvmobile']:
            # Get the corresponding kernel size
            k_size = int(op_string.split('x')[0])
            # Apply the operation
            output_block = self.inverted_mobile_bottleneck_block(input_x,
                                                                 kernel_size=k_size,
                                                                 expand_factor=6,
                                                                 n_filters=self.n_filters,
                                                                 strides=stride,
                                                                 layer_name=name)
        # If no operation is found then raise an error
        else:
            raise ValueError('Unrecognized operation passed to apply_operation')
        return output_block

    def squeeze_excitation_layer(self, input_x, ratio=2, n_filters=64,
                                 strides=1, layer_name='S&E'):
        '''
        Method used to apply a squeeze and excite layer to the given input.
        Params:
            - input_x: Tensor corresponding to the input.
            - ratio: Integer used to compute how many filters are in the squeeze
                     and excitation operation.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            if strides == 2:
                # Rescale input in order to allow application of squeezeing,
                # maintaining the same number of channels
                input_x = self.conv_layer(input_x,
                                          kernel_size=1,
                                          n_filters=n_filters,
                                          strides=strides,
                                          layer_name=layer_name+'_preprocess')
            # Get number of channels of the input (rescaled or not)
            input_dim = input_x.shape[-1]
            squeeze = GlobalAveragePooling2D()(input_x)
            excitation = Dense(units=input_dim / ratio,
                               name=layer_name + '_fully_connected1')(squeeze)
            excitation = Activation('relu',
                                    name=layer_name + '_relu')(excitation)
            excitation = Dense(units=input_dim,
                               name=layer_name + '_fully_connected2')(excitation)
            excitation = Activation('sigmoid')(excitation)
            excitation = tf.reshape(excitation, [-1, 1, 1, input_dim])
            #print('Input shape: {}\nEscitation shape: {}'.format(input_x.shape,
            #                                                     excitation.shape))
            scale = input_x * excitation
            #print('S&E Convolution shape: {}'.format(scale.shape))
            return scale

    def conv_layer(self, input_x, n_filters=64, kernel_size=3,
                   strides=1, layer_name='CONV'):
        '''
        Method used to apply a convolutional layer to the given input.
        Params:
            - input_x: Tensor corresponding to the input.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            # Apply convolution, followed by batch norm and relu
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                strides=strides,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_convolution')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            #print('Convolution shape: {}'.format(out_middle.shape))
            return out_middle

    def conv_max_layer(self, input_x, n_filters=64, kernel_size=3,
                       strides=1, layer_name='CONV_MAX'):
        '''
        Method used to apply a convolutional layer followed by max pooling to
        the given input.
        Params:
            - input_x: Tensor corresponding to the input.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            # Apply convolution followed by batch norm, relu and max pooling
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                strides=strides,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_convolution')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            out_middle = MaxPooling2D(pool_size=(2, 2),
                                      strides=1,
                                      padding='same',
                                      name=layer_name + '_mp')(out_middle)
            #print('Convolution+Max shape: {}'.format(out_middle.shape))
            return out_middle

    def conv_average_layer(self, input_x, n_filters=64, kernel_size=3,
                           strides=1, layer_name='CONV_AVG'):
        '''
        Method used to apply a convolutional layer followed by average average pooling
        to the given input.
        Params:
            - input_x: Tensor corresponding to the input.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            # Apply convolution, batch norm, relu and average pooling
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                strides=strides,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_convolution')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            out_middle = AveragePooling2D(pool_size=(2, 2),
                                          strides=1,
                                          padding='same',
                                          name=layer_name + '_mp')(out_middle)

            #print('Convolution+Average shape: {}'.format(out_middle.shape))
            return out_middle

    def dilation_conv_layer(self, input_x, n_filters=64, kernel_size=3,
                            dilation=2, strides=1, layer_name='DIL_CONV'):
        '''
        Method used to apply a dilated convolutional layer to the given input.
        Params:
            - input_x: Tensor corresponding to the input.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            # Apply dilated convolution, batch norm and relu
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                dilation_rate=dilation,
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_convolution')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            # If striding is needed we can use max pooling
            # to half the input height and width
            if strides == 2:
                out_middle = MaxPooling2D(pool_size=(2, 2),
                                          strides=strides,
                                          padding='same',
                                          name=layer_name + '_mp')(out_middle)
            #print('Dilation Convolution shape: {}'.format(out_middle.shape))
            return out_middle

    def depthwise_conv_layer(self, input_x, n_filters=64, kernel_size=3,
                             strides=1, layer_name='DEPTH_CONV'):
        '''
        Method used to apply a depthwise separable convolutional layer to the input.
        Params:
            - input_x: Tensor corresponding to the input.
            - n_filters: Integer defining the number of filters to use if the
                       input needs to be brought to a specific shape
                       (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                     If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        with tf.name_scope(layer_name):
            # Apply depthwise separable convolution, batch norm and relu
            out_middle = SeparableConv2D(n_filters,
                                         (kernel_size, kernel_size),
                                         padding='same',
                                         strides=strides,
                                         kernel_initializer="he_normal",
                                         depthwise_regularizer=regularizers.l2(self.alpha),
                                         name=layer_name + '_convolution')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            #print('Depthwise Convolution shape: {}'.format(out_middle.shape))
            return out_middle

    def shuffle_conv_layer(self, input_x, num_groups=4, n_filters=64,
                           kernel_size=3, strides=1, layer_name='SHUFFLE_CONV'):
        '''
        Method used to apply a depthwise separable convolutional layer to the input.
        Params:
            - input_x: Tensor corresponding to the input.
            - num_groups: Integer defining the number of groups used during
                          split while shuffling.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        # Get size of groups in which the input's filters are split
        sz = input_x.shape[3] // num_groups
        #print('Sz: {}'.format(sz))
        #print('Strides: {}'.format(strides))
        with tf.name_scope(layer_name):
            # 1x1 Group Convolution
            # print('Group convolution...')
            conv_side_layers = [
                Conv2D(n_filters // num_groups,
                       (1, 1),
                       padding='same',
                       strides=1,
                       kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(self.alpha),
                       name='{}_first_group_convolution_{}'.format(layer_name, i)) \
                       (input_x[:, :, :, i * sz:i * sz + sz])
                for i in range(num_groups)]
            #print('Concatenating group convolution...')
            out_middle = tf.concat(conv_side_layers, axis=-1)
            #print('Group convolution output shape:{}'.format(out_middle.shape))
            out_middle = BatchNormalization(name=layer_name + '_first_bn')(out_middle)
            out_middle = Activation('relu',
                                    name=layer_name + '_first_relu')(out_middle)
            # Channels Shuffle
            n, h, w, c = out_middle.get_shape().as_list()
            out_middle = tf.reshape(out_middle, shape=tf.convert_to_tensor(
                [tf.shape(out_middle)[0], h, w, num_groups, c // num_groups]))
            out_middle = tf.transpose(out_middle, tf.convert_to_tensor([0, 1, 2, 4, 3]))
            out_middle = tf.reshape(out_middle,
                                    shape=tf.convert_to_tensor([tf.shape(out_middle)[0],
                                                                h, w, c]))
            # Depthwise Convolution
            out_middle = SeparableConv2D(n_filters,
                                         (kernel_size, kernel_size),
                                         padding='same',
                                         kernel_initializer="he_normal",
                                         depthwise_regularizer=regularizers.l2(self.alpha),
                                         name=layer_name + '_sep_convolution')(out_middle)
            out_middle = BatchNormalization(name=layer_name + '_second_bn')(out_middle)
            #print('Separable convolution output shape:{}'.format(out_middle.shape))
            # 1x1 Group Convolution
            sz = out_middle.shape[3] // num_groups
            conv_side_layers = [
                Conv2D(n_filters // num_groups,
                       (1, 1),
                       padding='same',
                       strides=1,
                       kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(self.alpha),
                       name='{}_second_group_convolution_{}'.format(layer_name, i)
                       )(out_middle[:, :, :, i * sz:i * sz + sz]) for i in range(num_groups)]
            out_middle = tf.concat(conv_side_layers, axis=-1)
            out_middle = BatchNormalization(name=layer_name + '_third_bn')(out_middle)
            #print('Group convolution output shape:{}'.format(out_middle.shape))
            if strides == 2:
                out_middle = MaxPooling2D(pool_size=(2, 2),
                                          strides=strides,
                                          padding='same',
                                          name=layer_name + '_mp')(out_middle)
            #print('Shuffle convolution output shape:{}'.format(out_middle.shape))
            return out_middle

    def inverted_mobile_bottleneck_block(self, input_x, expand_factor=2, n_filters=64,
                                         kernel_size=3, strides=1, layer_name='INV_MOBILE'):
        '''
        Method used to apply a depthwise separable convolutional layer to the input.
        Params:
            - input_x: Tensor corresponding to the input.
            - expand_factor: Integer used to compute the number of filters used
                             in the expansion layer of the block.
            - n_filters: Integer defining the number of filters to use if the
                         input needs to be brought to a specific shape
                         (substitute of identity).
            - kernel_size: Integer defining the size of the convolutional kernel.
            - strides: Integer defining the stride of the convolutional block.
                       If strides=2 then we're building a reduction cell.
            - layer_name: String identifying the name of the subblock.
        Returns:
            - Tensor of the output of the operation.
        '''
        # Get input channels number and compute expansion layer channels
        input_channels = input_x.shape[3]
        expand_dims = expand_factor * input_channels
        with tf.name_scope(layer_name):
            # 1x1 convolution
            out_middle = Conv2D(expand_dims,
                                (1,1),
                                strides=1,
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_IMB_first_1xconv')(input_x)
            out_middle = BatchNormalization(name=layer_name + '_IMB_first_1xconv_bn')(out_middle)
            out_middle = ReLU(6, name=layer_name + '_IMB_first_1xconv_relu6')(out_middle)
            # Depthwise convolution
            out_middle = DepthwiseConv2D((kernel_size,kernel_size),
                                         padding='same',
                                         kernel_initializer="he_normal",
                                         depthwise_regularizer=regularizers.l2(self.alpha),
                                         name=layer_name + '_IMB_depthconv')(out_middle)
            out_middle = BatchNormalization(name=layer_name + '_IMB_depthconv_bn')(out_middle)
            out_middle = ReLU(6, name=layer_name + '_IMB_depthconv_relu6')(out_middle)
            # 1x1 convolution
            out_middle = Conv2D(n_filters,
                                (1, 1),
                                strides=strides,
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(self.alpha),
                                name=layer_name + '_IMB_second_1xconv')(out_middle)
            out_middle = BatchNormalization(name=layer_name + '_IMB_second_1xconv_bn')(out_middle)
            # Scale input depending on the selected stride
            if strides == 2 or input_channels != out_middle.shape[3]:
                # Apply 1x1 to input
                return out_middle
            else:
                return tf.add_n([out_middle, input_x], name=layer_name + '_IMB_out')

    def get_output(self):
        '''
        Method used to get the output of the block.
        Returns:
            - Tensor containing the output of the block.
        '''
        return self.out

    def get_string_representation(self):
        '''
        Method used to get the string representation of the built block.
        Returns:
            - Dictionary containing, name, inputs and operations of the block.
        '''
        repr = {'ID': self.full_block_name.split('_')[-1],
                'in': self.input_strings,
                'ops': self.operations}
        return repr


class CellStructure():
    """
    Class that defines the structure of a cell of the neural network to build.
    During initialization is mandatory to pass the cell name, the input tensors
    (output of previous cells), and the search space. Other parameters are
    optional and depends on the cell implementation.
    """
    def __init__(self, ID, inputs_dict, search_space, n_blocks=5,
                 n_blocks_per_block=2, cell_settings=None, n_filters=64, stride=1):
        """
        Constructor of the class.
        Params:
            - ID: String identifying the cell univoquely.
            - inputs_dict: Dictionary containing the input tensors corresponding
                           to the IDs. If these inputs have different shapes than
                           we will apply a convolutional layer to obtain same shapes.
            - search_space: Search space object.
            - n_blocks: Integer defining the number of blocks to use for a single cell.
            - n_blocks_per_block: Integer defining the number of parallel blocks
                                  to be used for each block in the cell.
            - cell_settings: Dictionary containing the settings for the current cell.
                             This is used only if a specific structure for a cell
                             is needed, otherwise the cell is built randomly from
                             the search space.
            - n_filters: Integer number of filters to be used for each convolutional
                         filter of this cell.
            - stride: Integer tha can either be 1 or 2, indicating if stride-1 or
                      stride-2 is used.
        """
        # Define hyperparameters
        self.name = ID
        self.inputs_dictionary = inputs_dict.copy()
        self.possible_inputs = self.inputs_dictionary.copy()
        #print('Possible inputs: {}'.format(self.possible_inputs))
        self.search_space = search_space.copy()
        # Check consistency of the stride
        if stride != 1 and stride != 2:
            raise ValueError('The chosen value of stride for this cell is not valid!')
        self.n_filters = n_filters
        self.stride = stride
        # First check if cell settings are available or not. If they are available
        # build the cell correspondingly. Otherwise, randomly initialize them and
        # build the cell correspondingly.
        if cell_settings is not None:
            self.cell_settings = cell_settings
            # Check if input shapes are ok or they need to be adjust
            valid_input_shapes = check_inputs_shapes(self.inputs_dictionary)
            if not valid_input_shapes:
                # Various shapes might depend from the fact that an input
                # comes from a stride-2 cell
                self.adjust_inputs()
            # Get number of blocks and subblocks in order to build the cell
            self.n_blocks = len(cell_settings['blocks'])
            self.n_blocks_per_block = len(cell_settings['blocks'][0]['ops'])
            # Build the cell from its settings
            self.build_cell_from_settings()
        else:
            # Check if input shapes are ok or they need to be adjust
            valid_input_shapes = check_inputs_shapes(self.inputs_dictionary)
            if not valid_input_shapes:
                # Various shapes might depend from the fact that
                # an input comes from a stride-2 cell
                self.adjust_inputs()
            # Define number of blocks and subblocks in order to build the cell
            self.n_blocks = n_blocks
            self.n_blocks_per_block = n_blocks_per_block
            # Build the cell randomly
            self.cell_settings = self.generate_random_cell()

    def generate_random_cell(self):
        """
        Generate cell settings randomly, for each block we randomly select the inputs
        from the set of possible inputs and a set of operations that do not create issues
        with shapes.
        """
        # Define empty settings and extract possible inputs
        cell_settings = {'blocks': []}
        possible_inputs = self.inputs_dictionary.copy()
        possible_inputs_strings = [key for key, _ in self.inputs_dictionary.items()]
        # Define empty outputs and iterate over the number of blocks in the cell
        blocks_out = []
        for index in range(self.n_blocks):
            # For each block selects randomly n inputs,
            # where n is the number of subblocks per block
            if index == 0:
                # For the first block, at least one input should connect
                # to the latest cell. Otherwise a cell might be generated
                # but not be linked to the model.
                first_input_string = possible_inputs_strings[-1]
                other_input_string = random.choices(possible_inputs_strings,
                                                    k=self.n_blocks_per_block - 1)
                inputs_strings = [first_input_string] + other_input_string
            else:
                inputs_strings = random.choices(possible_inputs_strings, k=self.n_blocks_per_block)
            # Convert input strings into the corresponding tensors
            inputs_tensors = [possible_inputs[i] for i in inputs_strings]
            # Define the block name and try to build it
            block_name_full = 'cell_{}_block_{}'.format(self.name, index)
            valid = False
            while not valid:
                # If we are dealing with a striding cell, we use stride = 2
                # in the block construction only when the input of the block
                # is a cell output and not the output of another block.
                # So if self.stride == 2 and 'block' is in a string of
                # input_strings list, then pass stride == 1 to the block
                # we need to build.
                if self.stride == 2 and check_substring_in_list('block',
                                                                inputs_strings):
                    strides = get_strides_from_input_strings('block',
                                                             inputs_strings)
                    block = BlockStructure(full_block_name=block_name_full,
                                           search_space=self.search_space,
                                           inputs=inputs_tensors,
                                           input_strings=inputs_strings,
                                           operations=random.choices(\
                                                self.search_space.get_kernels(),
                                                k=self.n_blocks_per_block),
                                           n_filters=self.n_filters,
                                           strides_list=strides)
                else:
                    block = BlockStructure(full_block_name=block_name_full,
                                           search_space=self.search_space,
                                           inputs=inputs_tensors,
                                           input_strings=inputs_strings,
                                           operations=random.choices(\
                                                self.search_space.get_kernels(),
                                                k=self.n_blocks_per_block),
                                           n_filters=self.n_filters,
                                           strides_list=[self.stride for _ in \
                                                range(self.n_blocks_per_block)])
                # Get block validity and if not valid, delete it.
                # A new block will be built randomly if this one is not valid.
                valid = block.valid
                if not block.valid:
                    del block
            # Once a valid block is obtained append its output to the list of blocks.
            blocks_out.append(block.get_output())
            # Append the output of the new block to the list of possible inputs
            # for the next blocks to be built.
            possible_inputs_strings.append(block_name_full)
            possible_inputs[block_name_full] = block.get_output()
            cell_settings['blocks'].append(block.get_string_representation())
        # Produce the ouput of the cell as the single output of the block
        # or the concatenation of the outputs of the blocks.
        if len(blocks_out) == 1:
            self.cell_out = blocks_out[0]
        else:
            self.cell_out = Concatenate(name='output_cell_{}'\
                                        .format(self.name))(blocks_out)
        # Check cell's output shape, if it is not proper then build another
        # random cell. This is done to avoid building valid cell that have
        # strange shapes.
        if self.check_ouput_shape():
            self.possible_inputs = possible_inputs.copy()
            self.possible_inputs_strings = possible_inputs_strings.copy()
            return cell_settings
        else:
            cell_settings = self.generate_random_cell()
            return cell_settings

    def build_cell_from_settings(self):
        """
        Method that builds the cell from the settings passed to it.
        The cell_settings object has a predefined structure that allows
        easy cell building.
        """
        # Get settings of each block and get the possible inputs
        blocks_definitions = self.cell_settings['blocks']
        possible_inputs = self.inputs_dictionary.copy()
        possible_inputs_strings = [key for key, _ in self.inputs_dictionary.items()]
        # Define empty outputs and iterate over the definitions of the blocks
        blocks_out = []
        for index, block_def in enumerate(blocks_definitions):
            # Get block name, inputs and operations required
            block_id = block_def['ID']
            block_name_full = 'cell_{}_block_{}'.format(self.name, block_id)
            block_inputs_strings = block_def['in']
            block_operations_strings = block_def['ops']
            # Get input tensors from the dictionary of possible inputs
            block_inputs = [possible_inputs[i] for i in block_inputs_strings]
            # If we are dealing with a striding cell, we use stride = 2
            # in the block construction only when the input of the block
            # is a cell output and not the output of another block.
            # So if self.stride == 2 and 'block' is in a string of
            # input_strings list, then pass stride == 1 to the block
            # we need to build.
            if self.stride == 2 and check_substring_in_list('block',
                                                            block_inputs_strings):
                strides = get_strides_from_input_strings('block',
                                                         block_inputs_strings)
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=block_inputs,
                                       input_strings=block_inputs_strings,
                                       operations=block_operations_strings,
                                       n_filters=self.n_filters,
                                       strides_list=strides)
            else:
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=block_inputs,
                                       input_strings=block_inputs_strings,
                                       operations=block_operations_strings,
                                       n_filters=self.n_filters,
                                       strides_list=[self.stride for _ in \
                                            range(self.n_blocks_per_block)])
            # Once a valid block is obtained append its output to the list of blocks.
            blocks_out.append(block.get_output())
            # Append the output of the new block to the list of possible inputs
            # for the next blocks to be built.
            possible_inputs[block_name_full] = block.get_output()
            possible_inputs_strings.append(block_name_full)
        # Produce the ouput of the cell as the single output of the block
        # or the concatenation of the outputs of the blocks.
        if len(blocks_out) == 1:
            self.cell_out = blocks_out[0]
        else:
            self.cell_out = Concatenate(name='output_cell_{}'\
                                        .format(self.name))(blocks_out)
        # Check cell's output shape, if it is not proper then raise an error.
        # This is done to avoid building valid cell that have strange shapes.
        if self.check_ouput_shape():
            self.possible_inputs = possible_inputs.copy()
            self.possible_inputs_strings = possible_inputs_strings.copy()
        else:
            raise ValueError('The cell you tried to build is not valid!')

    def check_ouput_shape(self, addition=False):
        """
        Method that is invoked when the cell is finished building in order
        to check if its output has the proper shape.
        Params:
            - addition: Boolean that defines if a block is being added to the cell.
                        This is used while mutating a cell.
        Returns:
            - Boolean, True if proper shape (i.e. stride==1: h_o=h_i, w_o=w_i,
                                                             f_o=n_blocks*n_filters
                                                  stride==2: h_o=h_i/2, w_o=w_i/2,
                                                             f_o=n_blocks*n_filters).
                       False otherwise.
        """
        # Get output shapes
        output_shape = tuple(self.cell_out.get_shape().as_list())
        # If the list of output shapes is valid compare the final output shape
        # with the proper output shape
        if check_inputs_shapes(self.inputs_dictionary):
            input_shape = get_unique_input_shape(self.inputs_dictionary)
            if self.stride == 1:
                if addition:
                    proper_shape = tuple([input_shape[0],
                                          input_shape[1],
                                          input_shape[2],
                                          (self.n_blocks + 1) * self.n_filters])
                else:
                    proper_shape = tuple([input_shape[0],
                                          input_shape[1],
                                          input_shape[2],
                                          self.n_blocks * self.n_filters])
            elif self.stride == 2:
                if addition:
                    proper_shape = tuple([input_shape[0],
                                          math.ceil(input_shape[1] / 2),
                                          math.ceil(input_shape[2] / 2),
                                          (self.n_blocks + 1) * self.n_filters])
                else:
                    proper_shape = tuple([input_shape[0],
                                          math.ceil(input_shape[1] / 2),
                                          math.ceil(input_shape[2] / 2),
                                          self.n_blocks * self.n_filters])
            else:
                raise ValueError('The stride selected for this cell is not valid!')
            # If output shape matches the proper shape return True.
            if output_shape == proper_shape:
                return True
            else:
                return False
        else:
            raise ValueError('Input shapes are not valid since they do not coincide!')

    def adjust_inputs(self):
        """
        Method used to adjust inputs shape by adding 1x1 convolution and
        2x2 max-pooling to the inputs with higher dimension.
        """
        inputs_shapes_dictionary, smallest_shape = get_inputs_shape_dictionary(\
                                                        self.inputs_dictionary)
        # For each input in the input dictionary...
        for key, input in self.inputs_dictionary.items():
            input_shape = tuple(input.get_shape().as_list())
            # ...check if it is not the smallest shape (stride-wise)...
            if input_shape != smallest_shape:
                # ...if so, apply input scaling to this input to make it
                # of the same shape of the smallest input
                new_input = self.apply_input_scaling(input,
                                                     key,
                                                     input_shape,
                                                     smallest_shape)
                # Substitute the adjusted input in the inputs dictionary
                self.inputs_dictionary[key] = new_input

    def apply_input_scaling(self, input, key, input_shape, smallest_shape):
        """
        Method used to apply 1x1 convolution to the given input Tensor,
        in order to obtain a new input Tensor having a desired shape.
        Params:
            - input: Tensor that identifies the input that needs to be adjusted.
            - key: Name of the cell whose output is used as input and needs
                   to be adjusted.
            - input_shape: Shape of the input Tensor.
            - smallest_shape: Shape that is necessary to achieve.
        Returns:
            - Tensor obtained apply convolution to input.
        """
        # Define the adjusting layer name
        layer_name = key + '_adjust_to_cell_' + self.name
        with tf.name_scope(layer_name):
            # Check if width/height of the input is the same of the smallest...
            if input_shape[1] != smallest_shape[1]:
                # ...if not, it means that there is an input from a stride-2
                # cell and an input from a non stride-2 cell. Adjust this with
                # a convolutional layer with stride-2 over the input.
                out_middle = Conv2D(smallest_shape[3],
                                    (1, 1),
                                    padding='same',
                                    strides=2,
                                    name=layer_name + '_convolution')(input)
            else:
                # ...if so, only the number of filter is mismatching
                out_middle = Conv2D(smallest_shape[3],
                                    (1, 1),
                                    padding='same',
                                    strides=1,
                                    name=layer_name + '_convolution')(input)
            out_middle = BatchNormalization(name=layer_name + '_bn')(out_middle)
            out_middle = Activation('relu', name=layer_name + '_relu')(out_middle)
            return out_middle

    def add_new_block(self, block_definition=None):
        '''
        Method used to add a new block to an existing cell (used during
        cell mutation). A block can be added randomly or from its definition.
        Params:
            - block_definition: Dictionary identifying the structure of
                                the block to add.
        '''
        # Either use the block definition or add a new one randomly
        if block_definition is not None:
            self.add_new_block_from_definition(block_definition)
        else:
            self.cell_settings = self.add_new_random_block()

    def add_new_block_from_definition(self, block_definition):
        '''
        Method used to add a block to the cell from the definition of the block.
        Params:
            - block_definition: Dictionary that defines the block structure.
                                It should be of the form of block settings.
        '''
        # Get block's identity, inputs and outputs
        block_id = block_definition['ID']
        block_name_full = 'cell_{}_block_{}'.format(self.name, block_id)
        block_inputs_strings = block_definition['in']
        block_operations_strings = block_definition['ops']
        # Get input tensors from the dictionary of possible inputs
        block_inputs = [self.possible_inputs[i] for i in block_inputs_strings]
        # If we are dealing with a striding cell, we use stride = 2
        # in the block construction only when the input of the block
        # is a cell output and not the output of another block.
        # So if self.stride == 2 and 'block' is in a string of
        # input_strings list, then pass stride == 1 to the block
        # we need to build.
        if self.stride == 2 and check_substring_in_list('block',
                                                        block_inputs_strings):
            strides = get_strides_from_input_strings('block',
                                                     block_inputs_strings)
            block = BlockStructure(full_block_name=block_name_full,
                                   search_space=self.search_space,
                                   inputs=block_inputs,
                                   input_strings=block_inputs_strings,
                                   operations=block_operations_strings,
                                   n_filters=self.n_filters,
                                   strides_list=strides)
        else:
            block = BlockStructure(full_block_name=block_name_full,
                                   search_space=self.search_space,
                                   inputs=block_inputs,
                                   input_strings=block_inputs_strings,
                                   operations=block_operations_strings,
                                   n_filters=self.n_filters,
                                   strides_list=[self.stride for _ in \
                                        range(self.n_blocks_per_block)])
        # Once the block is constructed, obtain the list of previous blocks'
        # outputs and append the new block's output to the list.
        blocks_out = self.get_all_blocks_output()
        blocks_out.append(block.get_output())
        self.possible_inputs[block_name_full] = block.get_output()
        # Rewrite the ouput of the cell as the single output of the block
        # or the concatenation of the outputs of the blocks.
        if len(blocks_out) == 1:
            self.cell_out = blocks_out[0]
        else:
            self.cell_out = Concatenate()(blocks_out)
        # Update the number of blocks that are in the cell.
        self.n_blocks += 1
        # Check cell output's shape, if it is not proper then raise an error
        # because the block that is trying to be added does not fit to the cell.
        if not self.check_ouput_shape():
            raise ValueError('The block you tried to add does '
                             'not conform to the cell you selected')
        # Append the block definition to the dictionary of settings of the cell.
        self.cell_settings['blocks'].append(block.get_string_representation())

    def add_new_random_block(self):
        '''
        Method used to add a block to the cell randomly.
        '''
        # Copy actual cell settings to avoid overwriting them
        cell_settings = self.cell_settings.copy()
        # Get the smallest unused id for the new block
        block_id = self.get_new_id()
        # Get possible inputs for the new block
        possible_inputs = self.possible_inputs.copy()
        possible_inputs_strings = [key for key, _ in self.possible_inputs.items()]
        # Selects randomly the inputs for the new block
        inputs_strings = random.choices(possible_inputs_strings,
                                        k=self.n_blocks_per_block)
        inputs_tensors = [possible_inputs[i] for i in inputs_strings]
        # Define new block's name
        block_name_full = 'cell_{}_block_{}'.format(self.name, block_id)
        # Try adding a random block to the cell until we find a valid combination.
        valid = False
        while not valid:
            # If we are dealing with a striding cell, we use stride = 2
            # in the block construction only when the input of the block
            # is a cell output and not the output of another block.
            # So if self.stride == 2 and 'block' is in a string of
            # input_strings list, then pass stride == 1 to the block
            # we need to build.
            if self.stride == 2 and check_substring_in_list('block',
                                                            inputs_strings):
                strides = get_strides_from_input_strings('block',
                                                         inputs_strings)
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=inputs_tensors,
                                       input_strings=inputs_strings,
                                       operations=random.choices(\
                                            search_space.get_kernels(),
                                            k=self.n_blocks_per_block
                                            ),
                                       n_filters=self.n_filters,
                                       strides_list=strides)
            else:
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=inputs_tensors,
                                       input_strings=inputs_strings,
                                       operations=random.choices(\
                                            search_space.get_kernels(),
                                            k=self.n_blocks_per_block
                                            ),
                                       n_filters=self.n_filters,
                                       strides_list=[self.stride for _ in \
                                            range(self.n_blocks_per_block)])
            # Check if the built block is valid, if not delete it.
            valid = block.valid
            if not block.valid:
                del block
        # Once the block is constructed, obtain the list of previous blocks'
        # outputs and append the new block's output to the list.
        blocks_out = self.get_all_blocks_output()
        blocks_out.append(block.get_output())
        possible_inputs_strings.append(block_name_full)
        possible_inputs[block_name_full] = block.get_output()
        cell_settings['blocks'].append(block.get_string_representation())
        # Rewrite the ouput of the cell as the single output of the block
        # or the concatenation of the outputs of the blocks.
        if len(blocks_out) == 1:
            self.cell_out = blocks_out[0]
        else:
            self.cell_out = Concatenate(name='output_cell_{}'\
                                        .format(self.name))(blocks_out)
        # Check cell's output shape, if it is  proper update the number
        # of blocks, and the inputs. If it is not proper then build
        # another random cell.
        if self.check_ouput_shape(addition=True):
            self.n_blocks += 1
            self.possible_inputs = possible_inputs.copy()
            self.possible_inputs_strings = possible_inputs_strings.copy()
            return cell_settings
        else:
            cell_settings = self.add_new_random_block()
            return cell_settings

    def mutate_block(self, id_of_block_to_mutate, inps_p=0.5, ops_p=0.5):
        """
        Method used to mutate a block of the considered cell.
        Params:
            - id_of_block_to_mutate: String identifying the chosen block to mutate.
            - inps_p: Probability of mutating an input link.
            - ops_p: Probability of mutating an operation.
        """
        # Copy actual cell settings to avoid overwriting them.
        cell_settings = self.cell_settings.copy()
        # Get block to mutate from its string id.
        index_of_block_to_mutate = self.get_block_index_from_id(id_of_block_to_mutate)
        block_to_mutate = self.cell_settings['blocks'][index_of_block_to_mutate]
        # Get possible inputs and skimm them. New inputs can only be inputs
        # of previous blocks or cells to avoid looping or disconnection
        # from the network.
        possible_inputs = self.possible_inputs.copy()
        possible_inputs = remove_later_blocks_from_inputs(possible_inputs,
                                                          id_of_block_to_mutate)
        possible_inputs_strings = [key for key, _ in possible_inputs.items()]
        # Flip coins to mutate inputs and/or operations.
        mutate_inps = (random.uniform(0, 1) < inps_p)
        mutate_ops = (random.uniform(0, 1) < ops_p)
        # Get new or old inputs, depending if inputs need to be mutated.
        if mutate_inps:
            inputs_strings = random.choices(possible_inputs_strings,
                                            k=self.n_blocks_per_block)
            inputs_tensors = [possible_inputs[i] for i in inputs_strings]
        else:
            inputs_strings = block_to_mutate['in']
            inputs_tensors = [possible_inputs[i] for i in inputs_strings]
        # Get new or old operations, depending if ops need to be mutated.
        if mutate_ops:
            new_operations = random.choices(self.search_space.get_kernels(),
                                            k=self.n_blocks_per_block)
        else:
            new_operations = block_to_mutate['ops']
        # Define the name of the block (it should be the same previous name).
        block_name_full = 'cell_{}_block_{}'.format(self.name,
                                                    id_of_block_to_mutate)
        # Try mutating randomly the block until we find a valid combination.
        valid = False
        while not valid:
            # If we are dealing with a striding cell, we use stride = 2
            # in the block construction only when the input of the block
            # is a cell output and not the output of another block.
            # So if self.stride == 2 and 'block' is in a string of
            # input_strings list, then pass stride == 1 to the block
            # we need to build.
            if self.stride == 2 and check_substring_in_list('block',
                                                            inputs_strings):
                strides = get_strides_from_input_strings('block',
                                                         inputs_strings)
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=inputs_tensors,
                                       input_strings=inputs_strings,
                                       operations=new_operations,
                                       n_filters=self.n_filters,
                                       strides_list=strides)
            else:
                block = BlockStructure(full_block_name=block_name_full,
                                       search_space=self.search_space,
                                       inputs=inputs_tensors,
                                       input_strings=inputs_strings,
                                       operations=new_operations,
                                       n_filters=self.n_filters,
                                       strides_list=[self.stride for _ in \
                                            range(self.n_blocks_per_block)])
            # Check if the built block is valid, if not delete it.
            valid = block.valid
            if not block.valid:
                del block
        # Once the block is constructed, obtain the list of previous blocks'
        # outputs and append the new block's output to the list.
        blocks_out = self.get_all_blocks_output()
        blocks_out[index_of_block_to_mutate] = block.get_output()
        possible_inputs_strings.append(block_name_full)
        possible_inputs[block_name_full] = block.get_output()
        cell_settings['blocks'][index_of_block_to_mutate] = block.\
                                                    get_string_representation()
        # Rewrite the ouput of the cell as the single output of the block
        # or the concatenation of the outputs of the blocks.
        if len(blocks_out) == 1:
            self.cell_out = blocks_out[0]
        else:
            self.cell_out = Concatenate(name='output_cell_{}'\
                                        .format(self.name))(blocks_out)
        # Check cell's output shape, if it is not proper mutate randomly again.
        if self.check_ouput_shape(addition=False):
            self.possible_inputs = possible_inputs.copy()
            self.possible_inputs_strings = possible_inputs_strings.copy()
            self.cell_settings = cell_settings.copy()
        else:
            self.mutate_block(id_of_block_to_mutate)

    def get_new_id(self):
        '''
        Method used to get the smallest unused id value for a possible new block.
        Returns:
            - String of the new id
        '''
        last_block_string = self.cell_settings['blocks'][-1]
        last_block_id = last_block_string['ID']
        new_id = str(int(last_block_id) + 1)
        return new_id

    def get_block_index_from_id(self, id_of_block_to_find):
        '''
        Method used to find the index of a certain block from its id.
        Useful in case if blocks have weird IDs.
        Params:
            - id_of_block_to_mutate: ID of the block to find (String).
        Returns:
            - Integer indexing the found block or None.
        '''
        blocks_def = self.cell_settings['blocks']
        index = 0
        for block_def in blocks_def:
            block_id = block_def['ID']
            if block_id == id_of_block_to_find:
                return index
            index += 1
        return None

    def get_all_blocks_ids(self):
        '''
        Method used to obtain a list of all IDs of the blocks belonging to the cell.
        Returns:
            - List of blocks' IDs
        '''
        blocks_def = self.cell_settings['blocks']
        block_ids = []
        for block_def in blocks_def:
            block_ids.append(block_def['ID'])
        return block_ids.copy()

    def get_all_blocks_output(self):
        '''
        Method used to get all blocks outputs. Every time we add a block,
        we append its output to the list of possible future inputs,
        so we can simply grab that list and get the corresponding tensors.
        Returns:
            - List of Tensors identifying blocks' outputs.
        '''
        # For each entry in the inputs string dictionary...
        blocks_out = []
        for string_in in self.possible_inputs_strings:
            # ...check that it's a block output (not a cell output)...
            if 'block' in string_in:
                # ...and append the corresponding tensor to the list.
                blocks_out.append(self.possible_inputs[string_in])
        return blocks_out

    def get_cell_settings(self):
        '''
        Method to obtain the cell settings dictionary.
        '''
        return self.cell_settings

    def get_output(self):
        '''
        Method to obtain the cell output Tensor.
        '''
        return self.cell_out


class ModelBuilder():
    """
    Class that defines the structure of a neural network and builds it.
    During initialization is mandatory to pass the cells settings, the input
    tensor, the strides to use and the settings available in settings_parser.
    """
    def __init__(self, cells_settings, filters_list, strides_list,
                 settings, n_blocks=5, n_blocks_per_block=2):
        '''
        Initialization method. Parameters are stored and the model is built
        from the settings.
        Params:
            - cells_settings: List of cell settings that uniquely defines
                              the model structure.
            - filters_list: List of integer that defines the filters to
                            use in each cell of the model.
            - strides_list: List of integer that defines the strides value
                            to use in each cell of the model.
            - settings: Settings defined in the settings_parser file.
        '''
        # Store hyperparameters
        self.settings = settings
        self.filters_list = filters_list
        self.strides_list = strides_list
        # Build search space and model input
        self.search_space = SearchSpace(settings.search_space_path)
        self.input_shape = (settings.img_shape,
                            settings.img_shape,
                            settings.img_channels)
        self.model_input = keras.Input(shape=self.input_shape, name="img")
        # Build the model
        self.initialize(cells_settings, n_blocks, n_blocks_per_block)
        # Add global weight decay if specified
        if self.settings.weight_decay is not None:
            self.add_global_weight_decay()
        # Parameters needed for evolution
        self.to_train = True
        self.previous_name = None

    def initialize(self, cells_settings, n_blocks=5, n_blocks_per_block=2):
        '''
        Method used to build a model from its settings.
        Params:
            - cells_settings: List of cell settings that uniquely defines
                              the model structure.
            - n_blocks: Number of blocks to use in each cell. This is used
                        if cells settings are None (the model needs to be
                        built randomly).
            - n_blocks: Number of subblocks to use in each block. This is
                        used if cells settings are None (the model needs
                        to be built randomly).
        '''
        # Define empty cells list
        self.cells = []
        self.cells_descriptors = []
        self.inputs_dict = {'model_input': self.model_input}
        # Iterate over the cell settings
        for index, cell_settings in enumerate(cells_settings):
            # Build the cell from its settings
            self.cells.append(CellStructure(str(index),
                                            self.inputs_dict,
                                            self.search_space,
                                            cell_settings=cell_settings,
                                            n_blocks=n_blocks,
                                            n_blocks_per_block=n_blocks_per_block,
                                            n_filters=self.filters_list[index],
                                            stride=self.strides_list[index]))
            # Append its descriptor and its output to the model descriptor
            # and the future inputs respectively.
            self.cells_descriptors.append(self.cells[index].get_cell_settings())
            self.inputs_dict['cell_{}_out'.format(index)] = self.cells[index].\
                                                            get_output()
            # Define how many inputs will be kept for future cells
            # and remove older inputs.
            if 'model_input' in self.inputs_dict.keys():
                n_keys_to_keep = 1
            else:
                n_keys_to_keep = 2
            self.remove_shallow_outputs_from_input_dict(n_keys_to_keep=n_keys_to_keep)
        # After all cells have been added, get the last output
        # as the feature extractor output
        self.last_output = self.cells[-1].get_output()
        # Add the classification block and build the final keras model
        self.add_classification_layer()
        self.model = keras.Model(self.model_input,
                                 self.output,
                                 name='Model_from_cells')

    def add_classification_layer(self):
        '''
        Method used to add the classification block (i.e. global pooling
        and fully connected with softmax).
        '''
        global_pooling_out = GlobalAveragePooling2D(name=\
                    'classification_global_pooling')(self.last_output)
        self.output = Dense(self.settings.classes,
                            activation='softmax',
                            name='classification_out')(global_pooling_out)

    def add_global_weight_decay(self):
        '''
        Method used to add weight decay to all convolutional layers
        of the model after the model has already been built.
        '''
        # Iterate over each layer
        for layer in self.model.layers:
            # If it is a convolutional or fully connected layer
            # add kernel regularizer.
            if isinstance(layer, keras.layers.Conv2D) or \
                isinstance(layer, keras.layers.Dense):
                layer.add_loss(lambda: keras.regularizers.l2(
                                    self.settings.weight_decay)(layer.kernel))
            # Otherwise add bias regularizer.
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda: keras.regularizers.l2(
                                    self.settings.weight_decay)(layer.bias))

    def add_new_cell(self, cell_settings=None, filters=64, strides=1,
                     n_blocks=5, n_blocks_per_block=2):
        '''
        Method used to add a new cell to the model. The cell can be
        added having certain structure or being a random cell,
        but it will always be added as the last cell.
        Params:
            - cell_settings: Dictionary that defines the settings of
                             the cell to be added to the model.
            - filters: Number of filters to use in convolution
                       operations of the new cell.
            - strides: Defines the stride of the new cell. Either 1 or 2.
            - n_block: Number of blocks that will be used to build
                       the new cell.
            - n_blocks_per_block: Number of subblocks that will be
                                  used to build the new cell.
        '''
        # Get a new name for the cell.
        new_cell_name = self.get_new_cell_name()
        # Build the new cell from settings and append it to the list
        # of cells that constitute the model.
        self.cells.append(CellStructure(new_cell_name,
                                        self.inputs_dict,
                                        self.search_space,
                                        cell_settings=cell_settings,
                                        n_blocks=n_blocks,
                                        n_blocks_per_block=n_blocks_per_block,
                                        n_filters=filters,
                                        stride=strides))
        # Append its descriptor and its output to the model descriptor
        # and the future inputs respectively.
        self.cells_descriptors.append(self.cells[-1].get_cell_settings())
        self.inputs_dict['cell_{}_out'.format(new_cell_name)] = self.cells[-1].get_output()
        # Define how many inputs will be kept for future cells
        # and remove older inputs.
        if 'model_input' in self.inputs_dict.keys():
            n_keys_to_keep = 1
        else:
            n_keys_to_keep = 2
        self.remove_shallow_outputs_from_input_dict(n_keys_to_keep=n_keys_to_keep)
        # Update model output since the last cell is new.
        self.last_output = self.cells[-1].get_output()
        self.add_classification_layer()
        self.model = keras.Model(self.model_input, self.output, name='Model_from_cells')

    def add_new_block_to_cell(self, cell_name, block_settings=None):
        '''
        Method used to add a new block to a specific cell of the model.
        Params:
            - cell_name: String identifying the name of the cell to
                         expand with the new block.
            - block_settings: Dictionary that defines the settings of
                              the block to be added.
        '''
        # Find the cell index corresponding to the passed name
        cell_index = self.get_cell_index_from_name(cell_name)
        cell_to_expand = self.cells[cell_index]
        # Invoke Cell method to add the block from its settings
        cell_to_expand.add_new_block(block_settings)
        # Reinitialize the model. This is done in order to update
        # links between cells and blocks. Otherwise the model is broken.
        self.initialize(self.cells_descriptors)

    def mutate_block_in_cell(self, cell_name, block_id=None):
        '''
        Method used to mutate a block in a cell. The cell must be specified,
        while the block can be picked randomly.
        Params:
            - cell_name: String identifying the name of the cell to mutate.
            - block_id: ID of the block to mutate. If None a random block
                        is mutated.
        '''
        # Get mutation probability from the settings
        link_mutation_prob = self.settings.links_mut_p
        operation_mutation_prob = self.settings.ops_mut_p
        # Get cell from its name
        cell_index = self.get_cell_index_from_name(cell_name)
        cell_to_mutate = self.cells[cell_index]
        # If the block id is None randomly selects a block to mutate.
        if block_id is None:
            all_ids = cell_to_mutate.get_all_blocks_ids()
            block_id = random.choice(all_ids)
        # Invoke the Cell method to mutate the block
        cell_to_mutate.mutate_block(block_id,
                                    link_mutation_prob,
                                    operation_mutation_prob)
        # Reinitialize the model. This is done in order to update
        # links between cells and blocks. Otherwise the model is broken.
        self.initialize(self.cells_descriptors)

    def update_cells_after(self, cell_index):
        '''
        Method used to update cells of the model after a certain index.
        This method is out of date since we always use initialize to quickly
        rebuild the model after a modification of a cell is made.
        Params:
            - cell_index: Index of the cell from which we need
                          to update the model.
        '''
        # Iterate over the cells after the passed index
        for index in range(cell_index, len(self.cells)):
            # Rebuild the cell from its previous settings
            self.cells[index] = CellStructure(str(index),
                                              self.cells[index].\
                                                    inputs_dictionary,
                                              self.search_space,
                                              cell_settings=self.cells[index].\
                                                    cell_settings,
                                              n_blocks=self.cells[index].n_blocks,
                                              n_blocks_per_block=self.cells[index].\
                                                    n_blocks_per_block,
                                              n_filters=self.filters_list[index],
                                              stride=self.strides_list[index])
        # Update model output since the last cell is new.
        self.last_output = self.cells[-1].get_output()
        self.add_classification_layer()
        self.model = keras.Model(self.model_input, self.output, name='Model_from_cells')

    def get_model(self):
        '''
        Method used to extract the keras model from the ModelBuilder object.
        '''
        return self.model

    def print_summary(self):
        '''
        Helper method to print the keras summary avoiding to write print.
        '''
        print(self.model.summary())

    def plot_model_to_file(self, plot_name):
        '''
        Method used to plot the model structure in the log folder
        where all models are printed.
        Params:
            - plot_name: Name of the file to be stored (not path).
        '''
        # Get path of the log and plots folder from settings
        log_path = os.path.join(os.getcwd(), self.settings.log_folder)
        plot_path = os.path.join(log_path, self.settings.plot_folder)
        # Make the folder if it doesn't exist
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        file_path = os.path.join(plot_path, plot_name)
        # Plot the model with its name
        plot_model(self.model, to_file=file_path,
                   show_shapes=False, show_layer_names=False)

    def remove_shallow_outputs_from_input_dict(self, n_keys_to_keep):
        '''
        Method used to remove shallow cells outputs from the list of
        possible inputs, depending on the parameter n_keys_to_keep.
        Params:
            - n_keys_to_keep: Number of inputs to keep for deeper cells to use.
        '''
        # Get the keys of the available inputs.
        all_keys = [key for key, _ in self.inputs_dict.items()]
        # Keys to keep are last n_keys_to_keep keys. All others are to remove.
        keys_to_remove = all_keys[:-n_keys_to_keep]
        # Remove them from inputs.
        for key in keys_to_remove:
            self.inputs_dict.pop(key)

    def set_not_trainable_cells(self, cell_index):
        '''
        Method used to set shallow cells (already trained) to be non trainable.
        Params:
            - cell_index: Index of the cell that is being searched. All
                          previous cells are going to be set as non trainable.
        '''
        # Build list of names of cells to be set as not trainable
        non_trainable_cells_list = ['cell_{}'.format(i) for i in range(cell_index)]
        # Iterate over each layer
        for layer in self.model.layers:
            # Find if this layer is a layer that needs to be set as not trainable
            to_train = True
            for cell_name in non_trainable_cells_list:
                if cell_name in layer.name and not 'adjust' in layer.name:
                    to_train = False
            # If not to train then set it to not trainable
            if not to_train:
                layer.trainable = False


    def get_model_descriptor(self):
        '''
        Method used to extract the model descriptor list from the ModelBuilder.
        '''
        copied_desc = copy.deepcopy(self.cells_descriptors)
        return copied_desc

    def get_new_cell_name(self):
        '''
        Method used to get the smallest unused cell name.
        This is used when a new cell is added.
        '''
        return str(int(self.cells[-1].name) + 1)

    def get_cell_index_from_name(self, name_to_find):
        '''
        Method used to get the index of the cell in the cells list
        from the name of the cell.
        Params:
            - name_to_find: String identifying the name of the cell to find.
        Returns:
            - Integer, index of the found cell in the self.cells list.
              None if no cell is found having the required name.
        '''
        index = 0
        for cell in self.cells:
            if cell.name == name_to_find:
                return index
            index += 1
        return None

    def set_previous_name(self, old_name):
        '''
        Method used to set the previous name of a model if this is cloned.
        This is helpful during evolution where we need to avoid training
        models twice.
        Params:
            - old_name: Name of the model which is being cloned.
        '''
        self.previous_name = old_name

    def get_previous_name(self):
        '''
        Method used to get the previous name of a model if this is cloned.
        This is helpful during evolution where we need to avoid training
        models twice.
        '''
        return self.previous_name

    def set_to_train(self, to_train=True):
        '''
        Method used to set the flag indicating if this model needs
        to be trained or not. This is helpful during evolution,
        where we need to avoid training models twice.
        Params:
            - to_train: Boolean defining if the model needs training or not.
        '''
        self.to_train = to_train

    def get_to_train(self):
        '''
        Method used to get the flag indicating if this model needs
        to be trained or not. This is helpful during evolution,
        where we need to avoid training models twice.
        '''
        return self.to_train

if __name__ == '__main__':
    # Import settings from arg_parse
    args = settings_parser.arg_parse()
    search_space = SearchSpace(args.search_space_path)
    # Define model input
    input_shape = (args.img_shape,
                   args.img_shape,
                   args.img_channels)
    model_input = keras.Input(shape=input_shape, name="img")

    # Cell settings and input dictionary examples:
    # cell_settings = {'blocks': [{'ID': '0', 'in': ['model_input', 'model_input'], 'ops': ['3xmax', '3xmax']},
    #                             {'ID': '1', 'in': ['model_input', 'cell_0_block_0'], 'ops': ['3xmax', '3xmax']},
    #                             {'ID': '2', 'in': ['model_input', 'cell_0_block_1'], 'ops': ['3xmax', '3xmax']}]}
    # input_dictionary = {'model_input': model_input}

    # Define hyperparameters
    n_cells = 3
    filt_list = [16,32,64]
    strides_list = [1,2,2]
    n_blocks = 3
    n_blocks_per_block = 1

    # Build the model with the helper
    #blockPrint()
    model_helper = ModelBuilder(cells_settings=[None for _ in range(n_cells)],
                                filters_list=filt_list,
                                strides_list=strides_list,
                                settings=args,
                                n_blocks=n_blocks,
                                n_blocks_per_block=n_blocks_per_block)
    model_descriptor = model_helper.get_model_descriptor()
    #enablePrint()
    print('\nModel descriptor: {}'.format(model_descriptor))
    # Save plot of model to file
    plot_name = 'model_scratch_before_mutate_block.png'
    model_helper.plot_model_to_file(plot_name)

    # Add new cell to model and try plot it
    #model_helper.add_new_cell()


    #for i in range(len(model_helper.cells)):
    #    print('\nIndex: {}'.format(i))
    #    print('Cell name: {}'.format(model_helper.cells[i].name))
    #    print('Cell object: {}'.format(model_helper.cells[i]))
    #    print('Inputs dictionary: {}\n'.format(model_helper.cells[i].inputs_dictionary))

    #model_helper.print_summary()
    #blockPrint()
    model_helper.mutate_block_in_cell(cell_name='1')
    #enablePrint()
    #model_helper.print_summary()
    #blockPrint()
    print('\nModel descriptor after: {}'.format(model_descriptor))
    #print('Model descriptor after add random block: {}'.format(model_helper.get_model_descriptor()))
    # Save plot of model to file
    plot_name = 'model_scratch_after_mutate_block.png'
    model_helper.plot_model_to_file(plot_name)


    # Try reconstruct model from cells settings
    print('\nReconstruct model without added cell...')
    print('Model descriptor: {}'.format(model_descriptor))
    #blockPrint()
    new_model_h = ModelBuilder(cells_settings=model_descriptor,
                               filters_list=filt_list,
                               strides_list=strides_list,
                               settings=args,
                               n_blocks=n_blocks,
                               n_blocks_per_block=n_blocks_per_block)
    #enablePrint()
    # Save plot of model to file
    print('\nPlotting reconstructed model...')
    plot_name = 'model_reconstructed.png'
    new_model_h.plot_model_to_file(plot_name)

    print('\nGetting memory footprint...')
    print('Memory footprint: {:.2f} MBs'.format(get_memory_footprint(new_model_h.get_model())))
