import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Exhaustive Neural Architecture Search')

    # Parameters used for network model
    parser.add_argument('--img_shape', default=32, type=int,
                        help="Width and height of the images in the dataset")
    parser.add_argument('--img_channels', default=3, type=int,
                        help="Number of channels of the images in the dataset")
    parser.add_argument('--classes', default=10, type=int,
                        help="Number of classes of the dataset considered")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Number of images used in a single batch")

    # Parameters used for training and retraining
    parser.add_argument('--n_epochs', default=20, type=int,
                        help="Number of epochs used to train a model")
    parser.add_argument('--lr_start', default=0.1, type=float,
                        help="Initial learning rate used for training")
    parser.add_argument('--lr_decay_epochs', default=10, type=int,
                        help="Number of epochs after which the learning rate decays")
    parser.add_argument('--lr_decay_factor', default=10, type=float,
                        help="Factor to which the learning rate decays. "
                             "LR_new = LR_old / LR_decay_factor")
    parser.add_argument('--momentum', default=0.99, type=float,
                        help="Momentum to be used for SGD training")
    parser.add_argument('--weight_decay', default=None, type=float,
                        help="Weight decay of convolutional layers. Typical value is 4e-5")
    parser.add_argument('--links_mut_p', default=0.5, type=float,
                        help="Probability of mutating input links of a block during a model evolution")
    parser.add_argument('--ops_mut_p', default=0.5, type=float,
                        help="Probability of mutating operations of a block during a model evolution")

    # Parameters used for paths to folders
    parser.add_argument('--log_folder', type=str, default='../log',
                        help='Path of folder where logs are stored')
    parser.add_argument('--search_space_path', type=str, default='../search_space.txt',
                        help="Path of text file containing search space dictionary")
    parser.add_argument('--models_folder', type=str, default='trained-models',
                        help="Path to model file")
    parser.add_argument('--tensorboard_folder', type=str, default='tensorboard/',
                        help="Path to tensorboard folder")
    parser.add_argument('--image_folder', type=str, default='images',
                        help="Path to folder containing various images")
    parser.add_argument('--plot_folder', type=str, default='plots',
                        help="Path to folder containing plots of CNNs")

    # Parameters used for generations and network evolution
    parser.add_argument('--pop_size', default=10, type=int,
                        help="Number of individuals that compose the population of a single generation.")
    parser.add_argument('--gen_per_cell', default=1, type=int,
                        help="Number of generations used to search a single cell.")
    parser.add_argument('--n_models_to_keep', default=3, type=int,
                        help="Number of models to be kept for each generation."
                             "These models will be evolved in order to build future generation.")
    parser.add_argument('--n_random_models_per_gen', default=2, type=int,
                        help="Number of models to be randomly add to a generation after killing"
                             "worst performing model. This value smooths evolution steepness.")
    parser.add_argument('--max_iterations', default=1000, type=int,
                        help="Maximum number of iterations used to try mutating previous best models"
                             "for the new generation.")


    # Parameters used for single model building
    parser.add_argument('--n_cells', default=5, type=int,
                        help="Number of cells that compose each model")
    parser.add_argument('--n_blocks_per_cell', default=3, type=int,
                        help="Number of blocks that compose each cell")
    parser.add_argument('--n_subblocks_per_block', default=1, type=int,
                        help="Number of sub-blocks that compose each block in each cell")
    parser.add_argument('--filters_list', nargs='*', default=[16, 32, 32, 64, 64],
                        help="Number of filters to be used in each cell")
    parser.add_argument('--strides_list', nargs='*', default=[1, 2, 1, 2, 1],
                        help="Strides values to be used in each cell")



    args = parser.parse_args()

    return args
