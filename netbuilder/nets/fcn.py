"""
@author Miquel Marti miquelmr@kth.se
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe


def add_fcn_branch(netspec, attach_layer, skip_source_layer, num_classes,
                   is_train):
    from lego.fcn import FCNAssembleLego

    fcn_params = dict(skip_source_layer=skip_source_layer,
                      num_classes=num_classes, is_train=is_train)
    FCNAssembleLego(fcn_params).attach(netspec, [netspec[attach_layer]])

    return netspec


def get_vgg_fcn(is_train=True):
    from lego.base import BaseLegoFunction
    from imagenet import VGGNet
    import math
    netspec = VGGNet().stitch()

    # TODO: Add extra layers
    return netspec


def get_resnet_fcn(params):
    from lego.base import BaseLegoFunction
    from imagenet import ResNet
    import math

    is_train = params['is_train']
    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    fc_layers = params['fc_layers']
    blocks = params['blocks']
    attach_layer = params['attach_layer']
    skip_source_layer = params['skip_source_layer']
    num_classes = params['num_classes']

    netspec = ResNet().stitch(is_train=is_train, source='tt',
                              main_branch=main_branch,
                              num_output_stage1=num_output_stage1,
                              fc_layers=fc_layers, blocks=blocks)

    # Remove pool layer at end of base network
    net = netspec.__dict__['tops']
    net.popitem('pool')

    netspec = add_fcn_branch(netspec, attach_layer,
                             skip_source_layer, num_classes, is_train)

    return netspec
