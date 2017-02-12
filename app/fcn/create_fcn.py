"""
@author Miquel Marti miquelmr@kth.se
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
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
    prototxt files with a dummy data layer that must be changed to accept
    the desired input.
    """)
parser.add_argument('-t', '--type', help="""Resnet or VGGnet""")
parser.add_argument('-o', '--output_folder',
                    help="""Train and Test prototxt will be generated
                    as train.prototxt and test.prototxt""")
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
parser.add_argument('-n', '--num_output_stage1',
                    help="""Number of filters in stage 1 of resnet""",
                    type=int, default=128)
parser.add_argument('--attach_layer',
                    help="""Name of layer where FCN branch will be attached""")
parser.add_argument('--skip_source_layer', nargs='+',
                    help="""Name of layer(s) from which skip conenctions
                    originate""")
parser.add_argument('-b', '--blocks', type=int, nargs='+',
                    help="""Number of Blocks in the 4 resnet stages""",
                    default=[3, 4, 6, 3])
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""",
                    required=True)
parser.add_argument('-c', '--num_classes',
                    help="""Number of classes in detection dataset""",
                    type=int, default=21)


if __name__ == '__main__':

    args = parser.parse_args()

    # source, main_branch, num_output_stage1, fc_layers, blocks
    res_params = dict(data_layer=args.data_layer,
                      main_branch=args.main_branch,
                      num_output_stage1=args.num_output_stage1,
                      blocks=args.blocks,
                      attach_layer=args.attach_layer,
                      skip_source_layer=args.skip_source_layer,
                      num_classes=args.num_classes)

    print "Creating FCN", args.type
    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=True)
    else:
        res_params['phase'] = 'train'
        res_params['data_dir'] = args.data_dir_train
        res_params['split'] = args.splits[0]
        netspec = get_resnet_fcn(res_params)

    fp = open(args.output_folder + '/train.prototxt', 'w')
    print >> fp, 'name: "fcn'+args.type+'"'
    print >> fp, netspec.to_proto()
    fp.close()

    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=False)
    else:
        res_params['phase'] = 'test'
        res_params['data_dir'] = args.data_dir_test
        res_params['split'] = args.splits[1],
        netspec = get_resnet_fcn(res_params)

    fp = open(args.output_folder + '/test.prototxt', 'w')
    print >> fp, 'name: "fcn-'+args.type+'"'
    print >> fp, netspec.to_proto()
    fp.close()
