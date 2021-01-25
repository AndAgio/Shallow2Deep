import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import blockPrint, enablePrint
from model_builder import ModelBuilder

def mobilenet(args):
    # Mobile Net like module
    filt_list =    [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
    strides_list = [1,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  1,  1,  1,  2,   1,   1,   1,   1]
    n_blocks = 1
    n_blocks_per_block = 1
    # Build the model with the helper
    blockPrint()
    cells_settings = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xinvmobilex1']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_3_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_4_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_5_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_6_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_7_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_8_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_9_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_10_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_11_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_12_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_13_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_14_out'], 'ops': ['3xinvmobile']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_15_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_16_out'], 'ops': ['3xinvmobile']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_17_out'], 'ops': ['1xconv']}]}]

    model_helper = ModelBuilder(cells_settings=cells_settings,
                                filters_list=filt_list,
                                strides_list=strides_list,
                                settings=args,
                                n_blocks=n_blocks,
                                n_blocks_per_block=n_blocks_per_block)

    enablePrint()

    return model_helper.get_model()


def vgg(args):
    # VGG like module
    filt_list =    [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
    strides_list = [1,  2,  1,   2,   1,   2,   1,   2,   1,    1]
    n_blocks = 1
    n_blocks_per_block = 1
    # Build the model with the helper
    blockPrint()
    cells_settings = [{'blocks': [{'ID': '0', 'in': ['model_input'], 'ops': ['3xconv']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_0_out'], 'ops': ['3xmax']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_1_out'], 'ops': ['3xconv']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_2_out'], 'ops': ['3xmax']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_3_out'], 'ops': ['3xconv']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_4_out'], 'ops': ['3xmax']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_5_out'], 'ops': ['3xconv']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_6_out'], 'ops': ['3xmax']}]},

                      {'blocks': [{'ID': '0', 'in': ['cell_7_out'], 'ops': ['1xconv']}]},
                      {'blocks': [{'ID': '0', 'in': ['cell_8_out'], 'ops': ['1xconv']}]}]

    model_helper = ModelBuilder(cells_settings=cells_settings,
                                filters_list=filt_list,
                                strides_list=strides_list,
                                settings=args,
                                n_blocks=n_blocks,
                                n_blocks_per_block=n_blocks_per_block)

    enablePrint()

    return model_helper.get_model()
