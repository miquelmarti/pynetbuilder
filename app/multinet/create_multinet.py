output_folder"""
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
from nets.multinet import get_resnet_multi

parser = ArgumentParser(description="""
    This script generates multi-task networks for Object detection and
    Segmentation based on different base networks. Writes to the selected
    folder the train and test prototxt files with a custom data layer that
    gives labels for both tasks.
    """)
parser.add_argument('-t', '--type', help="""'ResNet' only at the moment""")
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
parser.add_argument('--tasks', help="""Tasks to add: 'fcn', 'ssd', 'all' """,
                    default="all")

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
parser.add_argument('--min_dim', help="""Mininum dimension of input image""",
                    type=int, default=300)


if __name__ == '__main__':

    args = parser.parse_args()

    # Compute depth of base network
    if args.type == 'ResNet':
        layers_x_block = 2 if args.main_branch == "normal" else 3
        n_layers = sum(map(lambda x: x*layers_x_block, args.blocks)) + 2
    else:
        raise NotImplementedError("This type of base network is not supported")
    print args.type, n_layers

    num_gpus = len(args.gpu_list)
    if num_gpus > 0:
        solver_mode = P.Solver.GPU
    else:
        solver_mode = P.Solver.CPU
        num_gpus = 1
    iter_size = args.batch_size / (args.batch_size_per_device *
                                   num_gpus)
    print "Accum batch size: ", (iter_size * num_gpus *
                                 args.batch_size_per_device)
    print "Iteration size: ", iter_size
    print "Batch size x {} dev: ".format(num_gpus), args.batch_size_per_device

    # Count number of test images
    with open(args.name_size_file, 'r') as f:
        num_test_image = len(f.readlines())

    output_folder = os.path.abspath(ags.output_folder)
    # Create output directory if not existent, will overwrite previous content
    if not os.path.exists(args.):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, 'snapshots')):
        os.makedirs(os.path.join(output_folder, 'snapshots'))

    res_params = dict(main_branch=args.main_branch,
                      data_layer=args.data_layer,
                      num_output_stage1=args.num_output_stage1,
                      blocks=args.blocks,
                      extra_blocks=args.extra_blocks,
                      extra_num_outputs=args.extra_num_outputs,
                      mbox_source_layers=args.mbox_source_layers,
                      attach_layer=args.attach_layer,
                      num_classes=args.num_classes,
                      skip_source_layer=args.skip_source_layer,
                      tasks=args.tasks)

    # TRAIN NET
    if args.type == 'ResNet':
        res_params['phase'] = 'train'
        res_params['data_dir'] = args.data_dir_train
        res_params['batch_size_per_device'] = args.batch_size_per_device
        res_params['label_map_file'] = args.label_map_file
        res_params['name_size_file'] = args.name_size_file
        res_params['test_out_dir'] = args.test_out_dir
        netspec = get_resnet_multi(res_params)
        name = 'multiResnet'+str(n_layers)
    else:
        raise NotImplementedError("This type of base network is not supported")

    # TODO: Fix complexity computation
    # from tools.complexity import get_complexity
    # params, flops = get_complexity(netspec=netspec)
    # print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    # print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

    with open(os.path.join(output_folder, 'train.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-train"'
        print >> fp, netspec.to_proto()

    # Create solver
    solver_params = {}
    execfile("./config/solver.params", solver_params)
    solver_params = solver_params['solver_params']
    solver_params.update(
        dict(iter_size=iter_size, solver_mode=solver_mode))
    solver = caffe_pb2.SolverParameter(
                train_net=os.path.join(output_folder, 'train.prototxt'),
                test_net=[os.path.join(output_folder, 'test.prototxt')],
                snapshot_prefix=os.path.join(output_folder,
                                             'snapshots/train'),
                **solver_params)
    with open(os.path.join(output_folder, 'solver.prototxt'), 'w') as fp:
        print >> fp, solver

    # TEST NET
    if args.type == 'ResNet':
        res_params['phase'] = 'test'
        res_params['data_dir'] = args.data_dir_test
        res_params['batch_size_per_device'] = 1
        netspec = get_resnet_multi(res_params)
    else:
        raise NotImplementedError("This type of base network is not supported")

    with open(os.path.join(output_folder, 'test.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-test"'
        print >> fp, netspec.to_proto()

    # Create solver_score
    solver_params.update(
        dict(iter_size=1, snapshot=0, max_iter=0,
             test_initialization=True, snapshot_after_train=False,
             test_iter=[num_test_image]))
    solver = caffe_pb2.SolverParameter(
            train_net=os.path.join(output_folder, 'train.prototxt'),
            test_net=[os.path.join(output_folder, 'test.prototxt')],
            snapshot_prefix=os.path.join(output_folder,
                                         'snapshots/train'),
            **solver_params)
    with open(os.path.join(output_folder,
              'solver_score.prototxt'), 'w') as fp:
        print >> fp, solver

    # DEPLOY NET
    if args.type == 'ResNet':
        res_params['phase'] = 'deploy'
        netspec = get_resnet_multi(res_params)

    with open(os.path.join(output_folder, 'deploy.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-deploy"'
        print >> fp, netspec.to_proto()

    # Create train script
    with open(os.path.join(output_folder, 'train.sh'), 'w') as fp:
        print >> fp, 'DATE=`date +%Y-%m-%d_%H-%M-%S`'
        print >> fp, '$CAFFE_ROOT/build/tools/caffe train \\'
        print >> fp, '--solver="{}" \\'.format(
            os.path.join(output_folder, 'solver.prototxt'))
        if args.weights != '':
            print >> fp, '--weights="{}" \\'.format(os.path.abspath(
                args.weights))
        print >> fp, '--gpu {} 2>&1  | tee {}'.format(
            ",".join(str(x) for x in args.gpu_list),
            os.path.join(output_folder, "train_$DATE.log")
            )
