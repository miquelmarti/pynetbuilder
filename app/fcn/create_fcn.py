"""
@author Miquel Marti miquelmr@kth.se
"""

import sys
from argparse import ArgumentParser
import os

DATASETS_DIR = os.environ['DATASETS']
sys.path.append('netbuilder')

from tools.complexity import get_complexity
from nets.fcn import get_vgg_fcn, get_resnet_fcn


parser = ArgumentParser(description="""
    This script generates Fully Convolutional Networks for Semantic
    Segmentation based on VGGnet or Resnets with upsampling layers
    and skip connections. Write to the selected folder the train and test
    prototxt files with the selected data layer. Currently only available for
    PascalVOC but can be changed to accept the desired input.
    """)
parser.add_argument('-t', '--type', help="""Resnet or VGGnet""")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt
    will be generated as train.prototxt and test.prototxt""")

# Data params
# TODO: Load arguments from config file instead
parser.add_argument('-dl', '--data_layer', help="""Data layer to use""",
                    default="pascal")
parser.add_argument('-d', '--data_dir_train',
                    help="""Directory containing training data""",
                    default=os.path.join(DATASETS_DIR, 'sbdd/dataset'))
parser.add_argument('-D', '--data_dir_test',
                    help="""Directory containing test data""",
                    default=os.path.join(DATASETS_DIR, 'pascal/VOC2011'))
parser.add_argument('-s', '--splits',
                    help="""Name of splits for train and test""",
                    nargs="2", default=['train', 'seg11valid'])

# Train/test params
parser.add_argument('-g', '--gpu_list',
                    help="""List of gpus to use, CPU if empty""",
                    nargs="*", default=[0])
parser.add_argument('-bs', '--batch_size', help="""Total batch size""",
                    type=int, default=1)
parser.add_argument('--batch_size_per_device',
                    help="""Batch size per gpu""",
                    type=int, default=1)
parser.add_argument('--weights', help="""Weights file from which to start
    the training """, default='')

# Resnet params
parser.add_argument('-n', '--num_output_stage1', help="""Number of filters in
    stage 1 of resnet""", type=int, default=128)
parser.add_argument('-b', '--blocks', type=int, nargs='+', help="""Number of
    Blocks in the 4 resnet stages""", default=[3, 4, 6, 3])
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""",
                    required=True)

# FCN params
parser.add_argument('--attach_layer',
                    help="""Name of layer where FCN branch will be attached""",
                    required=True)
parser.add_argument('--skip_source_layer', nargs='*',
                    help="""Name of layer(s) from which skip conenctions
                    originate - Currently only works with one""")
parser.add_argument('-c', '--num_classes',
                    help="""Number of semantic classes in dataset""",
                    type=int, default=21)
parser.add_argument('--loss_normalize',
                    help="""Number of semantic classes in dataset""",
                    type=int, default=21)


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, 'snapshots')):
        os.makedirs(os.path.join(args.output_folder, 'snapshots'))

    res_params = dict(data_layer=args.data_layer,
                      main_branch=args.main_branch,
                      num_output_stage1=args.num_output_stage1,
                      blocks=args.blocks,
                      attach_layer=args.attach_layer,
                      skip_source_layer=args.skip_source_layer,
                      num_classes=args.num_classes)

    layers_x_block = 2 if args.main_branch == "normal" else 3
    n_layers = sum(map(lambda x: x*layers_x_block, args.blocks)) + 2
    print args.type, n_layers

    num_gpus = len(args.gpu_list)
    iter_size = args.batch_size / (args.batch_size_per_device *
                                   num_gpus)
    print "Accum batch size: ", (iter_size * num_gpus *
                                 args.batch_size_per_device)
    print "Iteration size: ", iter_size
    print "Batch size x {} gpu: ".format(num_gpus), args.batch_size_per_device

    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=True)
        name = 'fcnVGGNet'
    else:
        res_params['phase'] = 'train'
        res_params['data_dir'] = args.data_dir_train
        res_params['split'] = args.splits[0]
        netspec = get_resnet_fcn(res_params)
        name = 'fcnResnet'+str(n_layers)

    with open(os.path.join(args.output_folder, 'train.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-train"'
        print >> fp, netspec.to_proto()

    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=False)
    else:
        res_params['phase'] = 'test'
        res_params['data_dir'] = args.data_dir_test
        res_params['split'] = args.splits[1]
        netspec = get_resnet_fcn(res_params)

    with open(os.path.join(args.output_folder, 'test.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-test"'
        print >> fp, netspec.to_proto()

    # DEPLOY NET
    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=False)
    else:
        res_params['phase'] = 'deploy'
        netspec = get_resnet_fcn(res_params)

    with open(os.path.join(args.output_folder, 'deploy.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-deploy"'
        print >> fp, netspec.to_proto()
