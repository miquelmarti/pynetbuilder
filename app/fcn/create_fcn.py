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
parser.add_argument('--fc_layers', dest='fc_layers', action='store_true')
parser.add_argument('--no-fc_layers', dest='fc_layers', action='store_false')
parser.set_defaults(fc_layers=False)
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""",
                    required=True)
parser.add_argument('-c', '--num_classes',
                    help="""Number of classes in detection dataset""",
                    type=int, default=21)


if __name__ == '__main__':

    args = parser.parse_args()

    # source, main_branch, num_output_stage1, fc_layers, blocks
    res_params = dict(main_branch=args.main_branch,
                      num_output_stage1=args.num_output_stage1,
                      blocks=args.blocks,
                      attach_layer=args.attach_layer,
                      skip_source_layer=args.skip_source_layer,
                      fc_layers=args.fc_layers,
                      num_classes=args.num_classes)

    print "Creating FCN", args.type
    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=True)
    else:
        res_params['is_train'] = True
        netspec = get_resnet_fcn(res_params)

    # from tools.complexity import get_complexity
    # params, flops = get_complexity(netspec=netspec)
    # print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    # print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

    fp = open(args.output_folder + '/train.prototxt', 'w')
    print >> fp, 'name: "fcn'+args.type+'"'
    print >> fp, netspec.to_proto()
    fp.close()

    if args.type == 'VGG':
        netspec = get_vgg_fcn(is_train=False)
    else:
        res_params['is_train'] = False
        netspec = get_resnet_fcn(res_params)

    fp = open(args.output_folder + '/test.prototxt', 'w')
    print >> fp, 'name: "fcn-'+args.type+'"'
    print >> fp, netspec.to_proto()
    fp.close()
