# Shallow2Deep
Exhaustive Neural Architecture Search (NAS) mechanism that limits Neural Networks (NN) complexity to make them more transparent and understandable, while obtain high performances.

## Setup
You should first install the required dependencies using `pip install -r requirements.txt`

## Examples of usage
It is possible to run Shallow2Deep in a single batch or multiple batches. Running it in a single batch will process all cells that compose a NN in a single pass. Therefore, this mode is suggested if the user is not constrained by machine time limitations. In order to run Shallow2Deep environment constrained by time limitations --i.e. Google Colab, Kaggle notebooks -- it is possible to set the `gen_per_batch` parameter. This parameter makes Shallow2Deep run only for `gen_per_batch` generations at the time---e.g. if `gen_per_batch=5` Shallow2Deep runs for 5 generations, stores the results in the log folder and then stops. Shallow2Deep can continue the search process from where it was stopped selecting `start="false"`.

Basic run of Shallow2Deep can be invoked using:
```
python main.py
```

### Parameters
A set of parameters can be specified to select NN depth, the complexity of the cells that compose it and many other parameters:
* `n_cells` specifies the number of cells that compose the NN architecture.
* `n_blocks_per_cell` identifies the number of blocks that build each cell .
* `filters_list` specifies the number of filters used in each cell. This list should have length equal to `n_cells`.
* `strides_list` specifies the stride value used in each cell. This list should have length equal to `n_cells`.
* `pop_size` selects the number of NN models that compose the population of each generation of the evolutionary search algorithm in Shallow2Deep.
* `n_models_to_keep` selects the number of NN models that survive each generation in the evolutionary search.
* `n_random_models_per_gen` identifies the number of NN models that are created randomly in each generation of the evolutionary search.
* `gen_per_cell` specifies the number of generations used to search each cell. The total number of generations that Shallow2Deep will need to complete is `gen_per_cell*n_cells`.
* `n_epochs` identifies the number of epochs used to train each NN model in the population of a generation.
* `lr_start` specifies the intial learning rate
* `lr_decay_epochs` selects the number of epochs after which the lr is is shrinked. If `lr_decay_epochs>n_epochs` than lr_start is used for the whole training of NN models.
* `lr_decay_factor` identifies the factor by which the leraning rate is decayed after `lr_decay_epochs` epochs---e.g. `lr_new = lr_old / lr_decay_factor`.
* `batch_size` specifies the size of the batches used during training.

Post process
python process_log.py --dataset="fashion" --log_folder="../log_fashion"
