import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ast
import random
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
# from plot_model import plot_model
from keras.utils import plot_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D, Activation, BatchNormalization, Concatenate
from keras.layers import AveragePooling2D, DepthwiseConv2D, GlobalAveragePooling2D
# Import from files
import settings_parser
from utils import get_out_shape, check_inputs_shapes, get_unique_input_shape
from utils import get_inputs_shape_dictionary, check_substring_in_list
from utils import get_strides_from_input_strings


class SearchSpace():
    def __init__(self, file):
        self.space = {}
        self.initialize(file)

    def add_key(self, key):
        if key in self.space.keys():
            raise ValueError('Passed key is already in the search space')
        self.space[key] = []

    def add_parameter(self, key, parameter):
        if key in self.space.keys() and parameter in self.space[key]:
            raise ValueError('Passed key and parameters is already in the search space')
        if key in self.space.keys() and type(parameter) == list:
            for par in parameter:
                self.space[key].append(par)
        elif key in self.space.keys():
            self.space[key].append(parameter)
        if key not in self.space.keys():
            self.space[key] = []
            self.space[key] = parameter

    def initialize(self, file_path):
        file = open(file_path, "r")
        contents = file.read()
        self.space = ast.literal_eval(contents)
        # print('Search space: {}'.format(self.space))

    def get_random_value(self, key):
        if key not in self.space.keys():
            raise ValueError('Passed key doesn\'t exists in the search space')
        return random.choice(self.space[key])

    def check_kernel(self, ker):
        if ker in self.space['kernels']:
            return True
        return False

    def get_kernels(self):
        return self.space['kernels']


class BlockStructure():
    def __init__(self, ID, search_space, possible_inputs=[0,-1], inputs=None, operations=None):
        self.block_name = ID
        self.block_code = []
        if inputs is not None and operations is not None:
            for input in inputs:
                self.block_code.append(input)
            for operation in operations:
                if not search_space.check_kernel(operation):
                    raise ValueError('Selected operation for this block is not in the search space!')
                self.block_code.append(operation)
        elif inputs is None and operations is not None:
            for _ in range(len(operations)):
                self.block_code.append(random.choice(possible_inputs))
            for operation in operations:
                if not search_space.check_kernel(operation):
                    raise ValueError('Selected operation for this block is not in the search space!')
                self.block_code.append(operation)
        elif inputs is not None and operations is None:
            for input in inputs:
                self.block_code.append(input)
            for _ in range(len(inputs)):
                self.block_code.append(search_space.get_random_value('kernels'))
        else:
            print('You didn\'t select any input nor operation to pass '
                  'to the BlockStructure. Initializing random '
                  'operations with 2 inputs.')
            for _ in range(2):
                self.block_code.append(random.choice(possible_inputs))
            for _ in range(2):
                self.block_code.append(search_space.get_random_value('kernels'))


class CellStructure():
    def __init__(self, search_space, ID=0, n_blocks=5, settings=None):
        self.name = ID
        self.search_space = search_space
        self.n_blocks = n_blocks
        self.blocks = []
        self.initialize(cell_settings=settings)

    def initialize(self, cell_settings=None):
        if isinstance(cell_settings, dict):
            # Check number of blocks in the settings dictionary to be
            # equal to self.n_blocks, otherwise raise an Error
            if len(cell_settings['IDs']) == self.n_blocks:
                possible_inputs = [0,1]
                for index in range(len(cell_settings['IDs'])):
                    if index > 0:
                        possible_inputs.append(cell_settings['IDs'][index-1])
                    self.blocks.append(BlockStructure(ID=cell_settings['IDs'][index],
                                                      search_space=self.search_space,
                                                      possible_inputs=possible_inputs,
                                                      inputs=cell_settings['inputs'][index],
                                                      operations=cell_settings['operations'][index]))
            else:
                raise ValueError('Number of blocks in CellStructure initialize doesn\'t match n_blocks')
        else:
            possible_inputs = [-1, 0]
            for index in range(1, self.n_blocks+1):
                if index > 1:
                    possible_inputs.append(index - 1)
                self.blocks.append(BlockStructure(ID=index,
                                                  search_space=self.search_space,
                                                  possible_inputs=possible_inputs))


class ModelBuilder():
    def __init__(self, cells, settings):
        self.input_shape = (settings.img_shape,
                            settings.img_shape,
                            settings.img_channels)
        model_input = keras.Input(shape=self.input_shape, name="img")
        blocks_outputs = {}
        for cell in cells:
            print('\n\n\nBuilding new cell...')
            for block in cell.blocks:
                print('\n\nBuilding new block...')
                print('Block ID: {} structure: {}'.format(block.block_name, block.block_code))
                name = str(cell.name) + '_' + str(block.block_name)
                blocks_outputs[name] = self.build_block(block, cell, blocks_outputs, model_input)
                print('Blocks_outputs: {}'.format(blocks_outputs))
            # Concatenate all outputs of all blocks belonging to the cell
            print('\nConcatenating outputs of blocks...')
            layers_to_concatenate = []
            for block in cell.blocks:
                name = str(cell.name) + '_' + str(block.block_name)
                layers_to_concatenate.append(blocks_outputs[name])
            cell_output = Concatenate()(layers_to_concatenate)
            last_output = cell_output
        self.model = keras.Model(model_input, last_output, name='Model_from_cells')
        print(self.model.summary())
        plot_model(self.model, to_file='C:/Users/AgiolAnd/Documents/Projects/shallow2deep/scripts/model_plot.png', show_shapes=True, show_layer_names=True)

    def build_block(self, block, cell, blocks_outputs, model_input):
        if cell.name == 0:
            print('If of cell.name == 0')
            if block.block_code[0] != 0 and block.block_code[0] != -1 and block.block_code[1] != 0 and block.block_code[1] != -1:
                input_left = blocks_outputs[str(cell.name) + '_' + str(block.block_code[0])]
                input_right = blocks_outputs[str(cell.name) + '_' + str(block.block_code[1])]
            else:
                if block.block_code[0] == 0 or block.block_code[0] == -1:
                    input_left = model_input
                else:
                    input_left = blocks_outputs[str(cell.name) + '_' + str(block.block_code[0])]
                if block.block_code[1] == 0 or block.block_code[1] == -1:
                    input_right = model_input
                else:
                    input_right = blocks_outputs[str(cell.name) + '_' + str(block.block_code[1])]
        elif cell.name == 1:
            print('If of cell.name == 1')
            if block.block_code[0] != -1 and block.block_code[1] != -1:
                input_left = blocks_outputs[str(cell.name) + '_' + str(block.block_code[0])]
                input_right = blocks_outputs[str(cell.name) + '_' + str(block.block_code[1])]
            else:
                if block.block_code[0] == -1:
                    input_left = model_input
                if block.block_code[1] == -1:
                    input_right = model_input
        else:
            print('Else')
            input_left = blocks_outputs[str(cell.name) + '_' + str(block.block_code[0])]
            input_right = blocks_outputs[str(cell.name) + '_' + str(block.block_code[1])]
        # Apply operation on the inputs and add them to get the output
        print('input_left name: {}'.format(input_left))
        print('input_right name: {}'.format(input_right))
        print('operation left: {}'.format(block.block_code[2]))
        print('operation right: {}'.format(block.block_code[3]))
        name = 'cell_' + str(cell.name) + '_block_' + str(block.block_name) + '_left_op_' + block.block_code[2]
        out_left = self.apply_operation(input_left, block.block_code[2], name)
        name = 'cell_' + str(cell.name) + '_block_' + str(block.block_name) + '_right_op_' + block.block_code[3]
        out_right = self.apply_operation(input_right, block.block_code[3], name)
        out = tf.add(out_left, out_right, name='out_block_'+str(cell.name) + '_' + str(block.block_name))

        return out


    def apply_operation(self, input_x, op_string, name):
        if op_string == 'identity':
            output_block = input_x
        elif op_string == '3xdepthwise':
            output_block = self.depthwise_conv_layer(input_x, kernel_size=3, layer_name=name)
        elif op_string == '5xdepthwise':
            output_block = self.depthwise_conv_layer(input_x, kernel_size=5, layer_name=name)
        elif op_string == '3xdilated':
            output_block = self.dilation_conv_layer(input_x, dilation=2, layer_name=name)
        elif op_string == '3xmax':
            output_block = self.conv_max_layer(input_x, layer_name=name)
        elif op_string == '3xaverage':
            output_block = self.conv_average_layer(input_x, layer_name=name)
        elif op_string == '3xsqueezeexicte':
            output_block = self.squeeze_excitation_layer(input_x, ratio=2, layer_name=name)
        elif op_string == '3xshuffle':
            output_block = self.shuffle_conv_layer(input_x, num_groups=2, layer_name=name)
        else:
            raise ValueError('Unrecognized operation passed to apply_operation')
        return output_block

    def squeeze_excitation_layer(self, input_x, ratio=2, layer_name='S&E'):
        input_dim = input_x.shape[-1]
        with tf.name_scope(layer_name):
            squeeze = GlobalAveragePooling2D()(input_x)
            excitation = Dense(units = input_dim / ratio,
                               name = layer_name + '_fully_connected1')(squeeze)
            excitation = Activation('relu',
                                    name=layer_name + '_relu')(excitation)
            excitation = Dense(units = input_dim,
                               name = layer_name + '_fully_connected2')(excitation)
            excitation = Activation('sigmoid')(excitation)
            excitation = tf.reshape(excitation, [-1, 1, 1, input_dim])
            scale = input_x * excitation
            print('\n\n\nScale: {}\n\n\n'.format(scale.shape))
            return scale

    def conv_max_layer(self, input_x, n_filters=64, kernel_size=3, strides=1, layer_name='CONV_MAX'):
        with tf.name_scope(layer_name):
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                strides = strides,
                                name=layer_name+'_convolution')(input_x)
            out_middle = Activation('relu',
                                    name=layer_name+'_relu')(out_middle)
            out_middle = BatchNormalization(name=layer_name+'_bn')(out_middle)
            out_middle = MaxPooling2D(pool_size=(2, 2),
                                      strides=strides,
                                      padding='same',
                                      name=layer_name+'_mp')(out_middle)

            return out_middle

    def conv_average_layer(self, input_x, n_filters=64, kernel_size=3, strides=1, layer_name='CONV_AVG'):
        with tf.name_scope(layer_name):
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                strides=strides,
                                name=layer_name+'_convolution')(input_x)
            out_middle = Activation('relu',
                                    name=layer_name+'_relu')(out_middle)
            out_middle = BatchNormalization(name=layer_name+'_bn')(out_middle)
            out_middle = AveragePooling2D(pool_size=(2, 2),
                                      strides=strides,
                                      padding='same',
                                      name=layer_name+'_mp')(out_middle)

            return out_middle

    def dilation_conv_layer(self, input_x, n_filters=64, kernel_size=3, dilation=2, layer_name='DIL_CONV'):
        with tf.name_scope(layer_name):
            out_middle = Conv2D(n_filters,
                                (kernel_size, kernel_size),
                                padding='same',
                                dilation_rate=dilation,
                                name=layer_name+'_convolution')(input_x)
            out_middle = Activation('relu',
                                    name=layer_name+'_relu')(out_middle)
            out_middle = BatchNormalization(name=layer_name+'_bn')(out_middle)

            return out_middle

    def depthwise_conv_layer(self, input_x, n_filters=64, kernel_size=3, layer_name='DEPTH_CONV'):
        with tf.name_scope(layer_name):
            out_middle = SeparableConv2D(n_filters, #Depthwise
                                         (kernel_size, kernel_size),
                                         padding='same',
                                         name=layer_name+'_convolution')(input_x)
            print('\n\n\n{}\n\n\n'.format(out_middle.shape))
            out_middle = Activation('relu',
                                    name=layer_name+'_relu')(out_middle)
            out_middle = BatchNormalization(name=layer_name+'_bn')(out_middle)

            return out_middle

    def shuffle_conv_layer(self, input_x, num_groups=4, n_filters=64, kernel_size=3, strides=1, layer_name='SHUFFLE_CONV'):
        sz = input_x.shape[3] // num_groups
        print('\n\nSz: {}'.format(sz))
        with tf.name_scope(layer_name):
            # 1x1 Group Convolution
            print('Group convolution...')
            conv_side_layers = [
                Conv2D(n_filters // num_groups,
                       (1,1),
                       padding='same',
                       strides=strides,
                       name='{}_first_group_convolution_{}'.format(layer_name,i))(input_x[:,:,:,i*sz:i*sz+sz]) for i in range(num_groups)]
            print('Concatenating group convolution...')
            out_middle = tf.concat(conv_side_layers, axis=-1)
            print('Group convolution output shape:{}'.format(out_middle.shape))
            out_middle = BatchNormalization(name=layer_name + '_first_bn')(out_middle)
            out_middle = Activation('relu',
                                    name=layer_name + '_first_relu')(out_middle)
            # Channels Shuffle
            n, h, w, c = out_middle.get_shape().as_list()
            out_middle = tf.reshape(out_middle, shape=tf.convert_to_tensor([tf.shape(out_middle)[0], h, w, num_groups, c // num_groups]))
            out_middle = tf.transpose(out_middle, tf.convert_to_tensor([0, 1, 2, 4, 3]))
            out_middle = tf.reshape(out_middle, shape=tf.convert_to_tensor([tf.shape(out_middle)[0], h, w, c]))
            # Depthwise Convolution
            out_middle = SeparableConv2D(n_filters,
                                         (kernel_size, kernel_size),
                                         padding='same',
                                         name=layer_name + '_sep_convolution')(out_middle)
            out_middle = BatchNormalization(name=layer_name + '_second_bn')(out_middle)
            # 1x1 Group Convolution
            conv_side_layers = [
                Conv2D(n_filters // num_groups,
                       (1, 1),
                       padding='same',
                       strides=strides,
                       name='{}_second_group_convolution_{}'.format(layer_name,i)
                       )(out_middle[:, :, :, i * sz:i * sz + sz]) for i in range(num_groups)]
            out_middle = tf.concat(conv_side_layers, axis=-1)
            out_middle = BatchNormalization(name=layer_name + '_third_bn')(out_middle)
            print('Shuffle convolution output shape:{}\n\n'.format(out_middle.shape))
            return out_middle