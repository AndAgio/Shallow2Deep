import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import copy
import matplotlib.pyplot as plt
import heapq
import operator
import random
import pickle
import shutil
from shutil import copyfile
import tensorflow as tf
import keras
from keras.utils import plot_model
# Imported from files
import settings_parser
from utils import *
from dataset_importer import *
from train_helper import *
from model_footprint import get_memory_footprint
from model_builder import ModelBuilder, SearchSpace


class History():
    def __init__(self, n_cells, gen_per_cell, pop_size):
        '''
        Initialization method. It intializes a dictionary containing the names of
        each model for each generation and an empty sub-dictionary that will be used
        to store model descriptor, accuracy and size.
        Params:
            - n_cells: Number of cells that are used in the model that we are searching.
            - gen_per_cell: Number of generations that are used to search each cell.
            - pop_size: Number of models that are trained for each generation.
        '''
        self.n_cells = n_cells
        self.gen_per_cell = gen_per_cell
        self.pop_size = pop_size
        self.history_dict = {}
        for gen_ind in range(self.gen_per_cell*self.n_cells):
            for model_ind in range(self.pop_size):
                name = 'gen_{}_model_{}'.format(gen_ind, model_ind+1)
                self.history_dict[name] = {'Descriptor': None, 'Accuracy': None, 'Size': None}

    def add_generation_models(self, population):
        '''
        Method used after a generation is generated in order to update
        the descriptors available inside history dictionary.
        Params:
            - population: Dictionary containing the population of the models
                          that should be added to history.
                          Population should be of the form {Name: ModelBuilderObject},
                          where name is gen_n_model_m.
        '''
        for name, model_helper in population.items():
            if name not in self.history_dict.keys():
                raise ValueError('Could not find the model name in the predefined history of models!')
            self.history_dict[name]['Descriptor'] = model_helper.get_model_descriptor()
            self.history_dict[name]['Size'] = model_helper.get_model().count_params() / 1e6

    def update_generation_accuracies(self, accuracies):
        '''
        Method used after a generation is trained in order to update the accuracy values
        of models in history dictionary.
        Params:
            - accuracies: Dictionary containing the accuracies of the models.
                          Accuracies should be of the form {Name: Float}, where name is gen_n_model_m.
        '''
        for name, acc in accuracies.items():
            if name not in self.history_dict.keys():
                raise ValueError('Could not find the model name in the history of models!')
            self.history_dict[name]['Accuracy'] = acc

    def check_model_descriptor(self, descriptor):
        '''
        Method used to check if a certain model descriptor is already present in the history.
        This will be used to check whenever a model is mutated, if the mutation has already
        been trained in history.
        Params:
            - descriptor: Descriptor of the model that needs to be checked. It should be
                          of the form of ModelBuilder.get_model_descriptor()
        '''
        for name, subdict in self.history_dict.items():
            if descriptor == subdict['Descriptor']:
                return True
        return False

    def get_model_accuracy_from_name(self, model_name):
        '''
        Method used to get the accuracy of a certain model from its name.
        Params:
            - model_name: string identifying the model. It should be of the form gen_n_model_m.
        Returns:
            - float or None: accuracy of the model already trained and found in history.
        '''
        for name, subdict in self.history_dict.items():
            if name == model_name:
                return subdict['Accuracy']
        return None

    def get_accuracies_of_population(self, population):
        '''
        Method used to get the accuracy of a whole population from its models' names.
        Params:
            - population: dictionary identifying the models of a population.
                          It should be of the form of Population().population_dictionary().
        Returns:
            - dict or None: accuracies of the models in the population.
                            It has form {gen_n_model_m: 0.9754}.
        '''
        accs_dict = {}
        for name, _ in population.items():
            if name in list(self.history_dict.keys()):
                accs_dict[name] = self.history_dict[name]['Accuracy']
            else:
                raise ValueError('Couldn\'t find a model of the population in history!')
        return accs_dict

    def check_model_in_history(self, model_descriptor):
        '''
        Method used to check if a model is already in the history, given its model descriptor
        Params:
            - model_descriptor: Descriptor of a model. It should have the form
                                of ModelBuilder().get_model_descriptor().
        Returns:
            - Boolean: True if the model is already in history, False otherwise.
        '''
        for _, subdict in self.history_dict.items():
            if subdict['Descriptor'] == model_descriptor:
                return True
        return False

    def get_latest_generation_run(self):
        '''
        Method used to find the last generation that has been run.
        The last generation is the last having a value for the accuracy.
        '''
        latest_generation = -1
        for model_name, subdict in self.history_dict.items():
            if subdict['Accuracy'] is not None:
                latest_generation = int(model_name.split('_')[1])
        return latest_generation

    def get_descriptors_of_generation(self, gen_index):
        '''
        Method used to extract the descriptors of the models belonging
        to a specific generation, indexed by gen_index.
        Params:
            - gen_index: Index of the generation that we want to reload.
        '''
        descriptors = {}
        for model_name, subdict in self.history_dict.items():
            if 'gen_{}'.format(gen_index) in model_name:
                name = 'model_{}'.format(model_name.split('_')[-1])
                descriptors[name] = subdict['Descriptor']
        return descriptors

class Population():
    def __init__(self, settings, restart=False, prev_h=None):
        '''
        Initialization method used to setup the first generation of the evolutionary population.
        Params:
            - settings: args contanining all the settings defined in settings_parser.py
        '''
        self.settings = settings
        self.search_space = SearchSpace(settings.search_space_path)
        # Define the generation number to be 1 and get the number
        # of generations for each cell of the network
        self.generation = 0
        self.gen_per_cell = settings.gen_per_cell
        self.pop_size = settings.pop_size
        self.cell_to_search = self.generation // self.gen_per_cell
        # Define history object that will track models of different generations
        self.history = History(settings.n_cells, settings.gen_per_cell, settings.pop_size)
        # Define hyperparameters for filters list and strides list
        self.filters_list = settings.filters_list
        self.strides_list = settings.strides_list
        self.n_cells = settings.n_cells
        self.n_blocks = settings.n_blocks_per_cell
        self.n_blocks_per_block = settings.n_subblocks_per_block
        # Initialize population or recove previous population
        if restart:
            self.recover_past(prev_h)
        else:
            shutil.rmtree(settings.log_folder, ignore_errors=True)
            self.initialize_population()

    def initialize_population(self):
        '''
        Method used to initialize the first population. The first population is made of models
        that have a first cell which is random and all other cells that are simple convolutional layers.
        '''
        print('Generation: {} -> Generating population...'.format(self.generation))
        # Define an empty dictionary that will contain the population builders.
        self.population_dictionary = {}
        # Define how the cells settings should look like for each model in the population.
        # The first cell is built randomly, all other cells are just 3xconv.
        cells_settings = [None]
        for i in range(0, self.n_cells-1):
            cells_settings.append({'blocks': [{'ID': '0',
                                               'in': ['cell_{}_out'.format(i)],
                                               'ops': ['3xconv']}]})
        # Initialize the name of the first model as 1 and update them
        index = 1
        # Build random models and check if they can be added to the population
        # up until the population reaches the right size
        while len(self.population_dictionary.keys()) < self.pop_size:
            # The name of each model is of format gen_n_model_m
            name = 'gen_{}_model_{}'.format(self.generation, index)
            # Build the model without printing anything
            new_model_h = ModelBuilder(cells_settings=cells_settings,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            # Get model descriptor and check that this model is not already in population
            model_descriptor = new_model_h.get_model_descriptor()
            # If model not in population add it and update index
            if not self.model_in_population(model_descriptor):
                self.population_dictionary[name] = new_model_h
                index += 1
            # Otherwise don't do anything
            else:
                pass
        # Plot population's models in the generation_0 log folder
        self.plot_models_of_generation()
        # Update history of the models
        self.history.add_generation_models(self.population_dictionary)

    def model_in_population(self, model_descriptor):
        '''
        Method used to check if a model with a certain structure is already available
        in the current population.
        Params:
            - model_descriptor: Descriptor of the model that we want to search.
                                It should have the form of ModelBuilder().get_model_descriptor()
        Returns:
            Boolean: True if model already in population, False otherwise.
        '''
        for _, model_helper in self.population_dictionary.items():
            if model_descriptor == model_helper.get_model_descriptor():
                return True
        return False

        '''
        # Alternative (should be slower)
        already_added_models_descriptors = [val.get_model_descriptor() \
                                            for val in self.population_dictionary.values()]
        if model_descriptor not in already_added_models_descriptors:
            return True
        return False
        '''

    def plot_models_of_generation(self):
        '''
        Method used to plot the structure of all models belonging
        to the current generation. The plot will be stored in the
        log_folder/plots_folder/generation_n
        '''
        log_path = os.path.join(os.getcwd(),
                                self.settings.log_folder)
        plot_path = os.path.join(log_path,
                                 self.settings.plot_folder,
                                 'generation_{}'.format(self.generation))
        # If it is the first time deleaing with generation_n
        # make the corresponding folder
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # For each model in the population, get the ModelBuilder
        # object and use its model in plot_model
        for name, model_helper in self.population_dictionary.items():
            file_name = 'model_{}.png'.format(name.split('_')[-1])
            file_path = os.path.join(plot_path, file_name)
            plot_model(model_helper.get_model(), to_file=file_path,
                       show_shapes=True, show_layer_names=True)

    def run_evolution(self, data):
        '''
        Method used to run the evolutionary algorithm on the population. It runs
        a certain amount of generations per each cell, in order to identify the
        best configuration for each cell.
        Params:
            - data: Tuple containing a train_generator and a validation_generator.
                    These are the data used to fit the models of a population and
                    get their performances.
        '''
        # Run n-1 generations with fitting, survival and mutation
        for generation in range(self.gen_per_cell * self.n_cells - 1):
            self.run_generation(data)
        # Run last generation with only fitting and survival
        self.run_generation(data, last_gen=True)
        # Store final best models and their diagrams.
        self.store_final_bests()

    def run_batched_evolution(self, data):
        '''
        Method used to run the evolutionary algorithm on the population via batches.
        It starts the evolution for some generation and then stops it.
        The evolution will then resume where it stopped the next time the function
        is called. It runs a certain amount of generations per each cell,
        in order to identify the best configuration for each cell.
        Params:
            - data: Tuple containing a train_generator and a validation_generator.
                    These are the data used to fit the models of a population and
                    get their performances.
        '''
        # Check if this batch of generations is the last one or not.
        last_batch = (self.generation + self.settings.gen_per_batch >= self.gen_per_cell * self.n_cells)
        if not last_batch:
            for generation in range(self.settings.gen_per_batch):
                self.run_generation(data)
        if last_batch:
            # Run n-1 generations with fitting, survival and mutation
            for generation in range(self.generation, self.gen_per_cell * self.n_cells):
                self.run_generation(data)
            # Run last generation with only fitting and survival
            self.run_generation(data, last_gen=True)
            # Store final best models and their diagrams.
            self.store_final_bests()
            # Delete file containing continuation settings
            file_path = os.path.join(self.settings.log_folder, 'continuation_settings')
            file = open(file_path, "rb")
            args = pickle.load(file)
            file.close()
            os.remove(file_path)
            # Create a new pickle and text file containing the settings of the evolution
            file_path = os.path.join(args.log_folder, 'completed_evolution_settings')
            file = open(file_path, "wb")
            pickle.dump(args, file)
            file.close()
            file_path = os.path.join(args.log_folder, 'completed_evolution_settings.txt')
            file = open(file_path, "wb")
            file.write(args)
            file.close()

    def run_generation(self, data, last_gen=False):
        '''
        Method used to run a single generation of the evolutionary algorithm used
        to find the best architecture. A single generation requires fitting the whole
        population (paying attention to which models have already been trained), get
        their accuracies, update history, update generation value, keep best models,
        and (if it's not the last generation) mutate the population to obtain the
        next generation.
        Params:
            - data: Tuple containing a train_generator and a validation_generator.
                    These are the data used to fit the models of a population
                    and get their performances.
            - last_gen: Boolean defining if population mutation is necessary or not.
        '''
        # Fit population
        train_gen, test_gen = data
        self.fit_population(train_gen, test_gen)
        # Get accuracies of all models belonging to the population and update history correspondingly
        accuracies_dict = self.get_accuracies(test_gen)
        self.history.update_generation_accuracies(accuracies_dict)
        # Keep only best models inside the population
        self.keep_best_models()
        # Update the history log file
        self.write_history_log()
        if not last_gen:
            # Update generation value
            self.generation += 1
            # Get new population for the next generation
            # Check first if we need to mutate the same cell or start evolving the next cell
            changing_cell = self.check_changing_cell()
            self.get_new_population(new_cell=changing_cell)
        # Update the history log file
        self.write_history_log()
        # TO BE DONE:
        # - Introduce criterion selection in get_best_models()

    def fit_population(self, train_gen, test_gen):
        '''
        Method used to fit all models belonging to the current population.
        The previous generation best models must not be retrained.
        Therefore, we need to copy their previous h5 files to the next
        generation folder, in order for them to be recovered also
        in future generations.
        '''
        # Train all models belonging to the population of this generation
        index_fitting = 1
        prefix_string = 'Generation: {} -> Fitting population: '.format(self.generation)
        for name, model_helper in self.population_dictionary.items():
            suffix_string = ' Model {}/{}'.format(index_fitting,
                                                  len(self.population_dictionary.keys()))
            #print_progress_bar(index_fitting,
            #                   len(self.population_dictionary.keys()),
            #                   prefix=prefix_string,
            #                   suffix=suffix_string,
            #                   length=30)
            print('{} -> {}...'.format(prefix_string, suffix_string))
            # Check if the model needs to be trained or not...
            if model_helper.get_to_train():
                # If so, get the model and train it
                model = model_helper.get_model()
                # Define optimizer, metrics and loss first
                optimizer = keras.optimizers.SGD(lr=self.settings.lr_start,
                                                 momentum=self.settings.momentum,
                                                 decay=0.0,
                                                 nesterov=False)
                metrics = ['accuracy'] #['accuracy', 'top_k_categorical_accuracy']
                loss = keras.losses.CategoricalCrossentropy()
                # Compile model using optimizer, metrics and loss
                model.compile(optimizer=optimizer,
                              metrics=metrics,
                              loss=loss)
                # Define path where to save best model and add to callbacks
                self.gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                              self.settings.models_folder,
                                                              'generation_{}'.format(
                                                                    self.generation))
                if not os.path.exists(self.gen_folder_trained_models):
                    os.makedirs(self.gen_folder_trained_models)
                model_name = 'model_{}'.format(name.split('_')[-1])
                model_path = os.path.join(self.gen_folder_trained_models,
                                          '{}_trained.h5'.format(model_name))
                callbacks = get_standard_callbacks(self.settings, model_path)
                # Fit model and get the final testing accuracy
                model.fit(train_gen,
                          validation_data=test_gen,
                          epochs=self.settings.n_epochs,
                          callbacks=callbacks,
                          verbose=1)
                model_helper.set_to_train(False)
            else:
                # If a model is a previous generation best, then it had already been trained.
                # We can simply copy the old h5 file in the new generation folder. So that we
                # will be able to recover its accuracy also in the next generation.
                old_name = model_helper.get_previous_name()
                # Get previous h5 file and copy it to the new generation folder
                old_generation = old_name.split('_')[1]
                old_gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                             self.settings.models_folder,
                                                             'generation_{}'.format(old_generation))
                old_model_name = 'model_{}'.format(old_name.split('_')[-1])
                old_model_path = os.path.join(old_gen_folder_trained_models,
                                              '{}_trained.h5'.format(old_model_name))
                # Define name and path of new h5 file
                self.gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                              self.settings.models_folder,
                                                              'generation_{}'.format(self.generation))
                if not os.path.exists(self.gen_folder_trained_models):
                    os.makedirs(self.gen_folder_trained_models)
                model_name = 'model_{}'.format(name.split('_')[-1])
                model_path = os.path.join(self.gen_folder_trained_models,
                                          '{}_trained.h5'.format(model_name))
                # Copy the old h5 file in the new generation folder, with the new name
                copyfile(old_model_path, model_path)
            # Update index
            index_fitting += 1

    def get_accuracies(self, test_gen):
        '''
        Method used to get the best accuracies of each model belonging to the population.
        To get the best accuracy of each model, we need to load the best models from their
        h5 files obtained while training.
        Params:
            - test_gen: Generator containing the data used to validate a model.
        Returns:
            - Dict containing the accuracies of all models of the population.
        '''
        # Define empty dictionary
        index_evaluating = 1
        prefix_string = 'Generation: {} -> Evaluating population: '.format(self.generation)
        accs_dict = {}
        for name, _ in self.population_dictionary.items():
            suffix_string = ' Model {}/{}'.format(index_evaluating,
                                                  len(self.population_dictionary.keys()))
            #print_progress_bar(index_evaluating,
            #                   len(self.population_dictionary.keys()),
            #                   prefix=prefix_string,
            #                   suffix=suffix_string,
            #                   length=30)
            print('{} -> {}...'.format(prefix_string, suffix_string))
            # Load trained model from h5 file
            model_name = 'model_{}'.format(name.split('_')[-1])
            model_path = os.path.join(self.gen_folder_trained_models, '{}_trained.h5'.format(model_name))
            model = keras.models.load_model(model_path)
            # Get accuracy from loaded model
            loss, acc = model.evaluate(test_gen, verbose=0)
            # Add accuracy to dict of accuracies
            accs_dict[name] = acc
            # Update index
            index_evaluating += 1
        return accs_dict

    def keep_best_models(self):
        '''
        Method used to keep models with the best accuracies from the population.
        The accuracy of each model is taken from the history.
        ########### TO BE DONE: Add criterion to select best models ##########
        '''
        print('Generation: {} -> Keeping best models...'.format(self.generation))
        # Get best models' names by accuracy.
        accs_dict = self.history.get_accuracies_of_population(self.population_dictionary)
        best_models = heapq.nlargest(self.settings.n_models_to_keep,
                                     accs_dict.items(),
                                     key=operator.itemgetter(1))
        best_models = dict(best_models)
        # Remove from the population all the models which are not bests
        index = 0
        while index < len(self.population_dictionary.keys()):
            name = list(self.population_dictionary.keys())[index]
            if name not in best_models.keys():
                del self.population_dictionary[name]
            else:
                index += 1

    def check_changing_cell(self):
        '''
        Method used in order to check if we need to start evolving the next cell or not.
        Generation here is considered to be already updated.
        Returns:
            - Boolean: True if the cell to search has been moved to next, False otherwise
        '''
        # If generation divided by the gen/cell period does not give any remainder,
        # we need to update the cell where we work.
        if self.generation % self.gen_per_cell == 0:
            self.cell_to_search = self.generation // self.gen_per_cell
            return True
        else:
            return False

    def get_new_population(self, new_cell=False):
        '''
        Method used to get the population for the next generation.
        If new_cell is False, we need only to mutate the best models
        and add some random structure (to increase search variability).
        If new_cell is True, we need to randomly create a population
        for the new cell, starting from the best models of the last cell.
        Params:
            - new_cell: Boolean that defines if we need to search the
                        same cell (False) or the next (True).
        '''
        print('Generation: {} -> Generating population...'.format(self.generation))
        if not new_cell:
            # First duplicate best models and add them to the new generation
            self.duplicate_previous_best_models()
            # Clone and mutate randomly best models added to the generation
            index = len(self.population_dictionary.keys()) + 1
            models_to_clone = self.population_dictionary.copy()
            index = self.mutate_best_models(models_to_clone, index)
            # Add random models to the population
            index = self.add_random_models(index)
            # Plot models of the new generation
            self.plot_models_of_generation()
            # Update history of the models
            self.history.add_generation_models(self.population_dictionary)
        else:
            # Deepen best models
            self.deepen_previous_best_models()
            index = len(self.population_dictionary.keys()) + 1
            # Fill population with random models
            self.add_random_models(index)
            # Plot models of the new generation
            self.plot_models_of_generation()
            # Update history of the models
            self.history.add_generation_models(self.population_dictionary)
        # For each model in the population set the non trainable cells,
        # which are those cells before the actual cell to search.
        for model_name, model_builder in self.population_dictionary.items():
            model_builder.set_not_trainable_cells(self.cell_to_search)
            #print(model_builder.model.summary())

    def duplicate_previous_best_models(self):
        '''
        Method used to duplicate the best models of last generation in the
        population of the current generation. Models need to be cloned since
        population_dictionary contains the address of object and it can't be copied.
        '''
        # Define an empty dictionary for the cloned population
        new_population_dictionary = {}
        # Iterate over the models that have survived in the last generation
        for index in range(1, len(self.population_dictionary.keys()) + 1):
            # Define new name for the cloned models and get their cells settings
            name = 'gen_{}_model_{}'.format(self.generation, index)
            old_name = list(self.population_dictionary.keys())[index - 1]
            cells_settings = self.population_dictionary[old_name].get_model_descriptor()
            # Build the cloned models using ModelBuilder from cells settings
            new_model_h = ModelBuilder(cells_settings=cells_settings,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            # Cloned models must not be trained and we set their last generation name
            new_model_h.set_to_train(False)
            new_model_h.set_previous_name(old_name)
            # Copy the weights of the previous best model to the new model
            old_generation = old_name.split('_')[1]
            old_gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                         self.settings.models_folder,
                                                         'generation_{}'.format(old_generation))
            old_model_name = 'model_{}'.format(old_name.split('_')[-1])
            #print('Previous best name: {}'.format(old_model_name))
            old_model_path = os.path.join(old_gen_folder_trained_models,
                                          '{}_trained.h5'.format(old_model_name))
            #print('Previous best path: {}'.format(old_model_path))
            #print('Previous best descriptor: {}'.format(cells_settings))
            #new_model_h.model.set_weights(old_model_path)
            old_model = keras.models.load_model(old_model_path)
            # Copy weights of all cells
            #layer_index = 0
            #for layer in new_model_h.model.layers:
            #    layer.set_weights(old_model.layers[layer_index].get_weights())
            #    layer_index += 1
            for layer in old_model.layers:
                new_model_h.model.get_layer(layer.name).set_weights(layer.get_weights())
            # Add the cloned models to the new dictionary
            new_population_dictionary[name] = new_model_h
        # Substitute the population dictionary with the new one
        self.population_dictionary = new_population_dictionary

    def mutate_best_models(self, models_to_clone, index):
        '''
        Method used to mutate the best models in order to create the new population
        for the current generation. The best models are mutated randomly only inside
        the cell that is searched in this batch of generations. Mutation can involve
        links and operations of a single block of the cell and are picked randomly,
        with a certain probability defined in settings_parser.
        Params:
            - models_to_clone: Dictionary (copy of the population containing only
                               best models) that is used to select randomly which
                               model to clone.
            - index: Integer identifying the model of the generation that is being
                     built with the current mutation.
        Returns:
            - Integer: Updated index value.
        '''
        # Initialize iteration to avoid infinite loop when history is saturated
        iteration = 1
        while len(self.population_dictionary.keys()) < self.pop_size - \
              self.settings.n_random_models_per_gen \
              and iteration <= self.settings.max_iterations:
            name = 'gen_{}_model_{}'.format(self.generation, index)
            # Randomly selects a model, build its clone and randomly mutate it
            old_name = random.choice(list(models_to_clone.keys()))
            cells_settings = models_to_clone[old_name].get_model_descriptor()
            new_model_h = ModelBuilder(cells_settings=cells_settings,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            new_model_h.mutate_block_in_cell(cell_name=str(self.cell_to_search))
            # Check if the model built from mutation is already in population or history
            model_descriptor = new_model_h.get_model_descriptor()
            # If not add it to population and udate index
            if not self.model_in_population(model_descriptor) and \
                    not self.history.check_model_in_history(model_descriptor):
                self.population_dictionary[name] = new_model_h
                index += 1
                # Copy the previous model weights to the new model where they can be copied
                old_model = models_to_clone[old_name].model
                # Copy weights of previous cells
                list_cells_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search)]
                list_cells_not_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search,
                                                                             self.n_cells)]
                for layer in old_model.layers:
                    condition = (check_list_of_substring(list_cells_to_copy, layer.name)) and\
                            (not check_list_of_substring(list_cells_not_to_copy, layer.name))
                    if condition:
                        new_model_h.model.get_layer(layer.name).set_weights(layer.get_weights())
            else:
                pass
            iteration += 1
        # If max iterations reached print a warning message
        if iteration > self.settings.max_iterations:
            print_warning('Maximum iterations reached while mutating the cell!')
        return index

    def add_random_models(self, index):
        '''
        Method used to add some random models in order to fill up the new population
        for the current generation. The number of random models to add is defined
        in settings_parser.
        Params:
            - index: Integer identifying the model of the generation that is being
                     built with the current mutation.
        Returns:
            - Integer: Updated index value.
        '''
        # Initialize iteration to avoid infinite loop when history is saturated
        iteration = 1
        while len(self.population_dictionary.keys()) < self.pop_size \
              and iteration <= self.settings.max_iterations:
            name = 'gen_{}_model_{}'.format(self.generation, index)
            # Randomly build a model with None cell settings for the current cell
            old_name = random.choice(list(self.population_dictionary.keys()))
            cells_settings = self.population_dictionary[old_name].get_model_descriptor()
            cells_settings[self.cell_to_search] = None
            new_model_h = ModelBuilder(cells_settings=cells_settings,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            # Check if the model built  is already in population or history
            model_descriptor = new_model_h.get_model_descriptor()
            # If not add it to population and udate index
            if not self.model_in_population(model_descriptor) and \
               not self.history.check_model_in_history(model_descriptor):
                self.population_dictionary[name] = new_model_h
                index += 1
                # Copy the previous model weights to the new model where they can be copied
                old_model = self.population_dictionary[old_name].model
                # Copy weights of previous cells
                list_cells_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search)]
                list_cells_not_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search,
                                                                             self.n_cells)]
                for layer in old_model.layers:
                    condition = (check_list_of_substring(list_cells_to_copy, layer.name)) and\
                            (not check_list_of_substring(list_cells_not_to_copy, layer.name))
                    if condition:
                        new_model_h.model.get_layer(layer.name).set_weights(layer.get_weights())
            else:
                pass
            iteration += 1
        # If max iterations reached print a warning message
        if iteration > self.settings.max_iterations:
            print_warning('Maximum iterations reached while adding random models!\n'
                          'Current population size is: {}'\
                          .format(len(self.population_dictionary.keys())))
        return index

    def deepen_previous_best_models(self):
        '''
        Method used to duplicate the best models of last generation in the
        population of the current generation, when the next cell will be search.
        Models need to be cloned since population_dictionary contains the address
        of object and it can't be copied. Best models are cloned up to the last cell
        to search and then are randomly setup for the new cell to search.
        '''
        # Define an empty dictionary for the cloned population
        new_population_dictionary = {}
        # Iterate over the models that have survived in the last generation
        for index in range(1, len(self.population_dictionary.keys()) + 1):
            # Define new name for the cloned models and get their cells settings
            name = 'gen_{}_model_{}'.format(self.generation, index)
            old_name = list(self.population_dictionary.keys())[index - 1]
            cells_settings = self.population_dictionary[old_name].get_model_descriptor()
            # Define random cell settings for the new cell to search
            cells_settings[self.cell_to_search] = None
            # Build the cloned models using ModelBuilder from cells settings
            new_model_h = ModelBuilder(cells_settings=cells_settings,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            # Add the cloned models to the new dictionary
            new_population_dictionary[name] = new_model_h
            # Copy the previous model weights to the new model where they can be copied
            old_generation = old_name.split('_')[1]
            old_gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                         self.settings.models_folder,
                                                         'generation_{}'.format(old_generation))
            old_model_name = 'model_{}'.format(old_name.split('_')[-1])
            old_model_path = os.path.join(old_gen_folder_trained_models,
                                          '{}_trained.h5'.format(old_model_name))
            old_model = keras.models.load_model(old_model_path)
            # Copy weights of previous cells
            list_cells_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search)]
            list_cells_not_to_copy = ['cell_{}'.format(i) for i in range(self.cell_to_search,
                                                                         self.n_cells)]
            for layer in old_model.layers:
                condition = (check_list_of_substring(list_cells_to_copy, layer.name)) and\
                        (not check_list_of_substring(list_cells_not_to_copy, layer.name))
                if condition:
                    new_model_h.model.get_layer(layer.name).set_weights(layer.get_weights())
        # Substitute the population dictionary with the new one
        self.population_dictionary = new_population_dictionary

    def write_history_log(self):
        '''
        Method used to write the history variable to a log file for each generation.
        '''
        # Define file name and open it. It is place in the log folder.
        file_name = 'history_log.txt'
        file_path = os.path.join(self.settings.log_folder, file_name)
        file = open(file_path, "w")
        # Iterate of history items and write them in the file
        for model_name, model_resume in self.history.history_dict.items():
            file.write(str(model_name) + ' -> '+ str(model_resume) + '\n\n')
        # Close file
        file.close()
        # Write also a pickle file for history to continue evolution if needed
        file_name = 'history_log_pickle'
        file_path = os.path.join(self.settings.log_folder, file_name)
        file = open(file_path, "wb")
        pickle.dump(self.history.history_dict, file)
        file.close()

    def store_final_bests(self):
        '''
        Method used to store the best models of the last generation
        in a folder by itself. We will both store the h5 files and
        plot the models structure to make it clear for us which
        models are best.
        '''
        # Define folder for best models
        log_path = os.path.join(os.getcwd(), self.settings.log_folder)
        plot_path = os.path.join(log_path, 'best_models/plots')
        # Make the corresponding folder
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # For each model in the population, get the ModelBuilder object
        # and use its model in plot_model
        for name, model_helper in self.population_dictionary.items():
            # Get model accuracy to use it in the name of the file
            model_acc = self.history.get_model_accuracy_from_name(name)
            file_name = 'model_{}_acc_{:.3f}.png'.format(name.split('_')[-1],
                                                         model_acc*100)
            # Plot best models structure first
            file_path = os.path.join(plot_path, file_name)
            plot_model(model_helper.get_model(), to_file=file_path,
                       show_shapes=False, show_layer_names=False)
            # Get previous h5 file and copy it to the best_models folder
            generation = name.split('_')[1]
            gen_folder_trained_models = os.path.join(self.settings.log_folder,
                                                     self.settings.models_folder,
                                                     'generation_{}'.format(generation))
            model_name = 'model_{}'.format(name.split('_')[-1])
            old_model_path = os.path.join(gen_folder_trained_models,
                                          '{}_trained.h5'.format(model_name))
            # Define name and path of new h5 file
            best_trained_models_folder = os.path.join(log_path,
                                                      'best_models/trained-models')
            if not os.path.exists(best_trained_models_folder):
                os.makedirs(best_trained_models_folder)
            file_name = 'trained_model_{}_acc_{:.3f}.h5'.format(name.split('_')[-1],
                                                                model_acc*100)
            model_path = os.path.join(best_trained_models_folder,
                                      file_name)
            # Copy the old h5 file in the new generation folder, with the new name
            copyfile(old_model_path, model_path)

    def recover_past(self, prev_history_path):
        '''
        Method used to restart the evolution from the previous history.
        This is useful since the evolution might take a lot of time and it
        may be split into different runs (i.e. from gen 1 to 5, then from
        gen 5 to 10, and so on.)
        Params:
            - prev_history: path to the history_log file containing history
                            up to the interruption.
        '''
        # Get previous history and set it as the history of evolution
        with open(prev_history_path, 'rb') as pickle_file:
            self.history.history_dict = pickle.load(pickle_file)
        pickle_file.close()
        print('History dict:\n{}'.format(self.history.history_dict))
        # Get the value for the last generation run and update it.
        self.generation = self.history.get_latest_generation_run()
        self.cell_to_search = self.generation // self.gen_per_cell
        # Load trained models of last generation.
        self.load_models_of_generation(self.generation)
        # Get to the next generation, where it is possible to resume evolution
        # Keep only best models inside the population
        self.keep_best_models()
        # Update generation value
        self.generation += 1
        # Get new population for the next generation
        # Check first if we need to mutate the same cell or start evolving the next cell
        changing_cell = self.check_changing_cell()
        self.get_new_population(new_cell=changing_cell)
        # Update the history log file
        self.write_history_log()

    def load_models_of_generation(self, gen_index):
        '''
        Method used to load in the population the models of a previously
        trained generation. This is used whenever we need to stop and
        restart the evolution.
        Params:
            - gen_index: Index of the generation where we
                         stopped evolution previously.
        '''
        # Get models descriptors for models in the last trained generation.
        descriptors = self.history.get_descriptors_of_generation(gen_index)
        # Build all models from the descriptor and load the
        # weights from the corresponding h5 files.
        print('Generation: {} -> Restoring population...'.format(self.generation))
        # Define an empty dictionary that will contain the population builders.
        self.population_dictionary = {}
        # Initialize the name of the first model as 1 and update them
        index = 1
        for name, descriptor in descriptors.items():
            print('\nName: {} -> Descriptor: {}'.format(name, descriptor))
            # The name of each model is of format gen_n_model_m
            name = 'gen_{}_model_{}'.format(self.generation, index)
            # Build the model without printing anything
            new_model_h = ModelBuilder(cells_settings=descriptor,
                                       filters_list=self.filters_list,
                                       strides_list=self.strides_list,
                                       settings=self.settings,
                                       n_blocks=self.n_blocks,
                                       n_blocks_per_block=self.n_blocks_per_block)
            self.population_dictionary[name] = new_model_h
            # Load the weights of the model from the h5 file.
            folder_trained_models = os.path.join(self.settings.log_folder,
                                                 self.settings.models_folder,
                                                 'generation_{}'.format(self.generation))
            model_name = 'model_{}'.format(index)
            model_path = os.path.join(folder_trained_models,
                                      '{}_trained.h5'.format(model_name))
            new_model_h.model.load_weights(model_path)
            # Update index
            index += 1


def print_all_models_weights(log_folder, mod_folder, out_file_path):
    with open(out_file_path, 'w') as file:
        for generation_index in range(10):
            for model_index in range(1,11):
                gen_folder_trained_models = os.path.join(log_folder,
                                                         mod_folder,
                                                         'generation_{}'.format(generation_index))
                model_name = 'model_{}'.format(model_index)
                model_path = os.path.join(gen_folder_trained_models,
                                          '{}_trained.h5'.format(model_name))
                model = keras.models.load_model(model_path)
                file.write('\n\nGeneration {} -> Model {}'.format(generation_index, model_index))
                for layer in model.layers:
                    file.write('{} -> {}'.format(layer.get_config(), layer.get_weights()))
    file.close()

if __name__ == '__main__':
    args = settings_parser.arg_parse()

    # Start evolution from scratch
    #my_population = Population(args)

    # Restart evolution from history
    #prev_history_path = os.path.join(args.log_folder, 'history_log_pickle')
    #my_population = Population(args, restart=True, prev_h=prev_history_path)

    out_file = os.path.join(args.log_folder, 'models weights.txt')
    print_all_models_weights(args.log_folder, args.models_folder, out_file)

    # Import cifar10
    #cifar10_data = get_generator_from_cifar(args, split_train=True, small=False)

    # Run single generation
    #my_population.run_generation(cifar10_data)
    # Run evolution
    #my_population.run_evolution(cifar10_data)
