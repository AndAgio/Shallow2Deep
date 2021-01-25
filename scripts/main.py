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
        my_population = Population(args, restart=True, prev_h=prev_history_path)
    else:
        raise ValueError('The settings for start is not a valid setting.'
                         'It can either be True/False, Yes/No or 1/0')

    #out_file = os.path.join(args.log_folder, 'models weights.txt')
    #print_all_models_weights(args.log_folder, args.models_folder, out_file)
    # Import data
    data = get_generator_from_cifar(args, split_train=True, small=True)
    #data = get_generator_from_mnist(args, split_train=True, small=False)
    #data = get_generator_from_fashion_mnist(args, split_train=True, small=False)

    # Run evolution
    #my_population.run_evolution(data)
    # Run batched evolution
    my_population.run_batched_evolution(data)

'''
if __name__ == '__main__':
    args = settings_parser.arg_parse()

    train_generator, test_generator = get_generator_from_cifar(args)


    #Try plotting an image from train and test
    plot_image_from_generator(train_generator, name=os.path.join(args.image_folder,'train.png'))
    plot_image_from_generator(test_generator, name=os.path.join(args.image_folder,'test.png'))

    use_sota_net = False

    if use_sota_net:
        model = mobilenet(args)
        # model = vgg(args)
    else:
        # Generate model using the model builder class
        n_cells = 7
        filt_list = [16, 16, 16,16, 16, 16, 16]
        strides_list = [1, 1, 1, 1, 1, 1, 1]
        n_blocks = 2
        n_blocks_per_block = 1
        cells_settings = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']},
                                      {'ID': '1', 'in': ['model_input'], 'ops': ['3xconv']}]},
                          {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xconv']},
                                      {'ID': '1', 'in': ['cell_0_out'], 'ops': ['3xconv']}]},
                          {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xconv']},
                                      {'ID': '1', 'in': ['cell_1_out'], 'ops': ['3xconv']}]},
                          None,
                          None,
                          None,
                          None]
        # Build the model with the helper
        print('Building random model with {} cells, '
              'each with {} blocks...'.format(n_cells, n_blocks))
        blockPrint()
        model_helper = ModelBuilder(cells_settings=cells_settings,#[None for _ in range(n_cells)],
                                    filters_list=filt_list,
                                    strides_list=strides_list,
                                    settings=args,
                                    n_blocks=n_blocks,
                                    n_blocks_per_block=n_blocks_per_block)
        model = model_helper.get_model()
        enablePrint()

    # Save plot of model to file
    plot_name = 'model_train.png'
    if not os.path.exists(args.plot_folder):
        os.makedirs(args.plot_folder)
    file_path = os.path.join(args.plot_folder, plot_name)
    keras.utils.plot_model(model, to_file=file_path,
                           show_shapes=True, show_layer_names=True)
    print(model.summary())

    # Get memory usage of the model
    footprint = get_memory_footprint(model)
    print('Model memory footprint: {:.2f} MBs'.format(footprint))


    # Train model
    my_optimizer = keras.optimizers.SGD(lr=args.lr_start,
                                        momentum=args.momentum,
                                        decay=0.0,
                                        nesterov=False)
    # my_optimizer = keras.optimizers.Adam(lr=args.lr_start)
    my_metrics = ['accuracy']
    # my_metrics = ['accuracy', 'top_k_categorical_accuracy']
    # Compile model first
    model.compile(optimizer=my_optimizer,
                  metrics=my_metrics,
                  loss=keras.losses.CategoricalCrossentropy())
    # Define callbacks
    lr_callback = CustomLearningRateScheduler(epochs=args.n_epochs,
                                              lr_start=args.lr_start,
                                              lr_decay_epochs=args.lr_decay_epochs,
                                              lr_decay_factor=args.lr_decay_factor)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.1,
                                                       patience=5,
                                                       min_lr=1e-6)
    lr_printer_callback = CustomLearningRatePrinter()
    # Define path where to save best model
    if not os.path.exists(args.models_folder):
        os.makedirs(args.models_folder)
    model_path = os.path.join(args.models_folder, 'trained_model.h5')
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                       monitor='val_accuracy',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                           min_delta=0.01,
                                                           patience=10)
    terminate_nan_callback = tf.keras.callbacks.TerminateOnNaN()
    # For tensorboard check if folder is available
    if not os.path.exists(args.tensorboard_folder):
        os.makedirs(args.tensorboard_folder)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_folder,
                                                 update_freq='epoch')


    model.fit(train_generator,
              validation_data=test_generator,
              epochs=args.n_epochs,
              callbacks=[#lr_callback,
                         lr_printer_callback,
                         save_callback,
                         #early_stop_callback,
                         terminate_nan_callback,
                         tb_callback],
              verbose=True)

    # Evaluate student on test dataset
    loss, acc = model.evaluate(test_generator)
    print("Model accuracy: {:5.2f}%".format(100 * acc))

    '''
