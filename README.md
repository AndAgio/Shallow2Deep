python evolution.py --batch_size=2048 --n_cells=5 --n_blocks_per_cell=3 -filters_list 16 32 32 64 64 -strides_list 1 2 1 2 1 --n_epochs=50 --lr_decay_epochs=50 --pop_size=100 --gen_per_cell=10 --n_models_to_keep=10 --n_random_models_per_gen=20

!zip -r ../zipped_log.zip ../log
from google.colab import files
files.download("../zipped_log.zip")

First run of the code:
python main.py --start="true" --gen_per_batch=1 --batch_size=2048 --n_cells=5 --n_blocks_per_cell=5 -filters_list 16 32 32 64 64 -strides_list 1 2 1 2 1 --n_epochs=20 --lr_decay_epochs=20 --pop_size=100 --gen_per_cell=10 --n_models_to_keep=10 --n_random_models_per_gen=20

Successive runs:
python main.py
