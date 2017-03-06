"""
@author Miquel Marti miquelmr@kth.se
"""

import sys
from argparse import ArgumentParser
import os

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

DATASETS_DIR = os.environ['DATASETS']
sys.path.append('netbuilder')

from tools.complexity import get_complexity
from nets.fcn_ssd import get_resnet_fcn_ssd

parser = ArgumentParser(description="""
    This script generates multi-task networks for Object detection and
    Segmentation based on different base networks. Writes to the selected
    folder the train and test prototxt files with a custom data layer that
    gives labels for both tasks.
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
                    default=os.path.join(
                        DATASETS_DIR,
                        'VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb'))
parser.add_argument('-D', '--data_dir_test',
                    help="""Directory containing test data""",
                    default=os.path.join(
                        DATASETS_DIR,
                        'VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb'))
parser.add_argument('--label_map_file', help="""Label map file""",
                    default=os.path.join(
                        DATASETS_DIR,
                        'VOCdevkit/VOC0712/labelmap_voc.prototxt'))
parser.add_argument('--name_size_file', help="""Name size file""",
                    default=os.path.join(
                        DATASETS_DIR,
                        'VOCdevkit/VOC0712/test_name_size.txt'))

# Train/test params
parser.add_argument('-g', '--gpu_list',
                    help="""List of gpus to use, CPU if empty""",
                    nargs="*", default=[0])
parser.add_argument('-bs', '--batch_size', help="""Total batch size""",
                    type=int, default=32)
parser.add_argument('--batch_size_per_device',
                    help="""Batch size per gpu""",
                    type=int, default=1)
parser.add_argument('--test_out_dir', help="""Directory for test results""",
                    default='')
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
                    help="""Name of layer where FCN branch will be attached""")
parser.add_argument('--skip_source_layer', nargs='+',
                    help="""Name of layer(s) from which skip conenctions
                    originate - Currently only works with one""")

# SSD params
parser.add_argument('--mbox_source_layers', nargs='+', help="""Names of layers
    where detection heads will be attached""")
parser.add_argument('--extra_blocks', type=int, nargs='*', help="""Number of
    extra Blocks to be attached to Detection network""", default=[3, 3])
parser.add_argument('--extra_num_outputs', type=int, nargs='+', help="""Number
    of outputs of extra blocks Detection network""", default=[1024, 1024])
parser.add_argument('-c', '--num_classes', help="""Number of classes in
    detection dataset""", type=int, default=21)


if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, 'snapshots')):
        os.makedirs(os.path.join(args.output_folder, 'snapshots'))

    res_params = dict(main_branch=args.main_branch,
                      data_layer=args.data_layer,
                      num_output_stage1=args.num_output_stage1,
                      blocks=args.blocks,
                      extra_blocks=args.extra_blocks,
                      extra_num_outputs=args.extra_num_outputs,
                      mbox_source_layers=args.mbox_source_layers,
                      attach_layer=args.attach_layer,
                      num_classes=args.num_classes,
                      skip_source_layer=args.skip_source_layer)

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

    with open(args.name_size_file, 'r') as f:
        num_test_image = len(f.readlines())

    # TRAIN NET
    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=True)
        name = 'multiVGGNet'
    else:
        res_params['phase'] = 'train'
        res_params['data_dir'] = args.data_dir_train
        res_params['batch_size_per_device'] = args.batch_size_per_device
        res_params['label_map_file'] = args.label_map_file
        res_params['name_size_file'] = args.name_size_file
        res_params['test_out_dir'] = args.test_out_dir
        netspec = get_resnet_fcn_ssd(res_params)
        name = 'multiResnet'+str(n_layers)

    # from tools.complexity import get_complexity
    # params, flops = get_complexity(netspec=netspec)
    # print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    # print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

    with open(os.path.join(args.output_folder, 'train.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-train"'
        print >> fp, netspec.to_proto()

    # Create solver
    solver_params = {}
    execfile("./config/solver.params", solver_params)
    solver_params = solver_params['solver_params']
    solver_params.update(
        dict(iter_size=iter_size, solver_mode=P.Solver.GPU))
    solver = caffe_pb2.SolverParameter(
                train_net=os.path.join(args.output_folder, 'train.prototxt'),
                test_net=[os.path.join(args.output_folder, 'test.prototxt')],
                snapshot_prefix=os.path.join(args.output_folder,
                                             'snapshots/train'),
                **solver_params)
    with open(os.path.join(args.output_folder, 'solver.prototxt'), 'w') as fp:
        print >> fp, solver

    # TEST NET
    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=False)
    else:
        res_params['phase'] = 'test'
        res_params['data_dir'] = args.data_dir_test
        res_params['batch_size_per_device'] = 1
        res_params['label_map_file'] = args.label_map_file
        res_params['name_size_file'] = args.name_size_file
        res_params['test_out_dir'] = args.test_out_dir
        netspec = get_resnet_fcn_ssd(res_params)

    with open(os.path.join(args.output_folder, 'test.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-test"'
        print >> fp, netspec.to_proto()

    # Create solver_score
    solver_params.update(
        dict(iter_size=1, solver_mode=P.Solver.CPU, snapshot=0, max_iter=0,
             test_initialization=True, snapshot_after_train=False,
             test_iter=[num_test_image]))
    solver = caffe_pb2.SolverParameter(
            train_net=os.path.join(args.output_folder, 'train.prototxt'),
            test_net=[os.path.join(args.output_folder, 'test.prototxt')],
            snapshot_prefix=os.path.join(args.output_folder,
                                         'snapshots/train'),
            **solver_params)
    with open(os.path.join(args.output_folder,
              'solver_score.prototxt'), 'w') as fp:
        print >> fp, solver

    # DEPLOY NET
    if args.type == 'VGG':
        netspec = get_vgg_ssdnet(is_train=False)
    else:
        res_params['phase'] = 'deploy'
        netspec = get_resnet_fcn_ssd(res_params)

    with open(os.path.join(args.output_folder, 'deploy.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-deploy"'
        print >> fp, netspec.to_proto()

    # Create train script
    with open(os.path.join(args.output_folder, 'train.sh'), 'w') as fp:
        print >> fp, 'DATE=`date +%Y-%m-%d_%H-%M-%S`'
        print >> fp, '$CAFFE_ROOT/build/tools/caffe train \\'
        print >> fp, '--solver="{}" \\'.format(
            os.path.join(os.path.abspath(
                args.output_folder), 'solver.prototxt')
            )
        if args.weights != '':
            print >> fp, '--weights="{}" \\'.format(os.path.abspath(
                args.weights))
        print >> fp, '--gpu {} 2>&1  | tee {}'.format(
            ",".join(str(x) for x in args.gpu_list),
            os.path.join(os.path.abspath(
                args.output_folder), "train_$DATE.log")
            )
