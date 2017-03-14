"""
@author Miquel Marti miquelmr@kth.se
"""


from argparse import ArgumentParser
import os

from caffe import params as P
from caffe.proto import caffe_pb2

from nets.multinet import get_resnet_multi


# TODO: Load arguments from config file instead

parser = ArgumentParser(description="""
    This script generates multi-task networks for Object detection and
    Segmentation based on different base networks. Writes to the selected
    folder the train and test prototxt files with a custom data layer that
    gives labels for both tasks.
    """)

parser.add_argument('-t', '--type', help="""'ResNet' only at the moment""",
                    default="ResNet")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt
    will be generated as train.prototxt and test.prototxt""",
                    default="")
parser.add_argument('--tasks', help="""Tasks to add: 'fcn', 'ssd', 'all' """,
                    default="all")

# Data params
parser.add_argument('-dl', '--data_layer', help="""Data layer to use""",
                    default="pascal")
parser.add_argument('-d', '--data_dir_train', help="""Directory containing
training data""", default="")
parser.add_argument('-D', '--data_dir_test', help="""Directory containing
test data""", default="")

# Data params for SSD
parser.add_argument('--label_map_file', help="""Label map file""",
                    default="")
parser.add_argument('--name_size_file', help="""Name size file""",
                    default="")
parser.add_argument('--num_test_image', help="""Num test images""", type=int,
                    default=0)
parser.add_argument('--min_dim', help="""Minimum dimension of input image""",
                    type=int, default=300)

# Train/test params
parser.add_argument('-g', '--gpu_list',
                    help="""List of gpus to use, CPU if empty""",
                    nargs="*", default=[])
parser.add_argument('-bs', '--batch_size', help="""Total batch size""",
                    type=int, default=32)
parser.add_argument('--batch_size_per_device',
                    help="""Batch size per gpu""",
                    type=int, default=1)
parser.add_argument('--test_out_dir', help="""Directory for test results""",
                    default='$WORK/test_results')
parser.add_argument('--weights', help="""Weights file from which to start
    the training """, default='')

parser.add_argument('--use_batchnorm', help="""Use Batch Normalization,
    if specified `use_global_stats` will be False so the batch statistics will
    be updated during training and the newly added blocks will include
    BatchNorm layers, otherwise the original BatchNorm layers will be kept but
    will use the previous statistics. Only makes sense when the batch size per
    device is greater than one.""", action='store_true')

# ResNet params
parser.add_argument('-n', '--num_output_stage1', help="""Number of filters in
    stage 1 of resnet""", type=int, default=128)
parser.add_argument('-b', '--blocks', type=int, nargs='+', help="""Number of
    Blocks in the 4 resnet stages""", default=[3, 4, 6, 3])
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""",
                    default='bottleneck')

# FCN params
parser.add_argument('--attach_layer',
                    help="""Name of layer where FCN branch will be
                    attached""", default='res5c_relu')
parser.add_argument('--skip_source_layer', nargs='*',
                    help="""Name of layer(s) from which skip connections
                    originate - Currently only works with one""",
                    default=['res4f_relu'])

# SSD params
parser.add_argument('--mbox_source_layers', nargs='+', help="""Names of layers
    where detection heads will be attached""", default=['res3d_relu',
                                                        'res4f_relu',
                                                        'res5c_relu',
                                                        'res6b_relu',
                                                        'res7b_relu',
                                                        'pool_last'])
parser.add_argument('--extra_blocks', type=int, nargs='*', help="""Number of
    extra Blocks to be attached to Detection network""", default=[2, 2])
parser.add_argument('--extra_num_outputs', type=int, nargs='*', help="""Number
    of outputs of extra blocks Detection network""", default=[1024, 1024])
parser.add_argument('-c', '--num_classes', help="""Number of classes in
    detection dataset""", type=int, default=21)


if __name__ == '__main__':

    args = parser.parse_args()

    # Compute depth of base network
    if args.type == 'ResNet':
        layers_x_block = 2 if args.main_branch == "normal" else 3
        n_layers = sum(map(lambda x: x*layers_x_block, args.blocks)) + 2
    else:
        raise NotImplementedError("This type of base network is not supported")

    print ""
    print "Creating", args.type, n_layers, "with", args.tasks.upper()
    print ""

    num_gpus = len(args.gpu_list)
    if num_gpus > 0:
        solver_mode = P.Solver.GPU
    else:
        solver_mode = P.Solver.CPU
        num_gpus = 1
    iter_size = args.batch_size / (args.batch_size_per_device *
                                   num_gpus)

    print "======================="
    print "Using ", "GPU" if solver_mode else "CPU"
    print "Accum batch size: ", (iter_size * num_gpus *
                                 args.batch_size_per_device)
    print "Iteration size: ", iter_size
    print "Batch size x {} dev: ".format(num_gpus), args.batch_size_per_device
    print "=======================\n"

    # Count number of test images
    if args.name_size_file is not "":
        if os.path.exists(args.name_size_file):
            with open(args.name_size_file, 'r') as f:
                num_test_image = len(f.readlines())
    elif args.num_test_image > 0:
        num_test_image = args.num_test_image
    else:
        num_test_image = 0
    if args.output_folder == "":
        output_folder = os.path.expandvars("$HOME/data")
    else:
        output_folder = os.path.expandvars(args.output_folder)

    # Create output directory if not existent, will overwrite previous content
    if not os.path.exists(output_folder):
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
                      tasks=args.tasks,
                      min_dim=args.min_dim,
                      num_test_image=num_test_image,
                      use_batchnorm=args.use_batchnorm)

    # TRAIN NET
    if args.data_dir_train == "":
        print "Warning! Data dir for train set not specified"
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

    print "Created {} - train".format(name)

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

    print "Created solver parameters file"

    # TEST NET
    if args.data_dir_test == "":
        print "Warning! Data dir for test set not specified"
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

    print "Created {} - test".format(name)

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

    print "Created solver_score parameters file"

    # DEPLOY NET
    if args.type == 'ResNet':
        res_params['phase'] = 'deploy'
        netspec = get_resnet_multi(res_params)
    else:
        raise NotImplementedError("This type of base network is not supported")

    with open(os.path.join(output_folder, 'deploy.prototxt'), 'w') as fp:
        print >> fp, 'name: "' + name + '-deploy"'
        print >> fp, netspec.to_proto()

    print "Created {} - deploy".format(name)

    # Create train script
    with open(os.path.join(output_folder, 'train.sh'), 'w') as fp:
        print >> fp, 'DATE=`date +%Y-%m-%d_%H-%M-%S`'
        print >> fp, '$CAFFE_ROOT/build/tools/caffe train \\'
        print >> fp, '--solver="{}" \\'.format(
            os.path.join(output_folder, 'solver.prototxt'))
        if args.weights != '':
            print >> fp, '--weights="{}" \\'.format(os.path.abspath(
                args.weights))
        if len(args.gpu_list) > 0:
            print >> fp, '--gpu {} 2>&1  | tee {}'.format(
                ",".join(str(x) for x in args.gpu_list),
                os.path.join(output_folder, "train_$DATE.log")
                )
        else:
            print >> fp, ' 2>&1  | tee {}'.format(
                os.path.join(output_folder, "train_$DATE.log")
            )

    print "Created train script"

    print ""
    print "Model ready to train!"
    print ""
