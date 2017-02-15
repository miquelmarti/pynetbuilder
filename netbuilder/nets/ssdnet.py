"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.

Modified by Miquel Marti miquelmr@kth.se
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe


def get_vgg_ssdnet(is_train=True):

    raise Exception('VGG not available for SSD')
    from lego.ssd import MBoxUnitLego, MBoxAssembleLego
    from lego.base import BaseLegoFunction
    from imagenet import VGGNet
    import math
    netspec = VGGNet().stitch()

    params = dict(base_network='VGGnet')
    last = SSDExtraLayersLego(params).attach(netspec, [netspec['fc7']])

    num_classes = 21
    min_dim = 300
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2',
                          'conv7_2', 'conv8_2', 'pool6']
    min_ratio = 20
    max_ratio = 95
    step = int(math.floor((max_ratio - min_ratio) /
                          (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    normalizations = [20, -1, -1, -1, -1, -1]

    assemble_params = dict(mbox_source_layers=mbox_source_layers,
                           normalizations=normalizations,
                           aspect_ratios=aspect_ratios,
                           num_classes=num_classes,
                           min_sizes=min_sizes,
                           max_sizes=max_sizes, is_train=is_train)
    MBoxAssembleLego(assemble_params).attach(netspec, [netspec['label']])

    return netspec


def get_resnet_ssdnet(params):
    from lego.data import VOCSegDataLego
    from lego.ssd import MBoxUnitLego, MBoxAssembleLego, SSDExtraLayersLego
    from lego.base import BaseLegoFunction
    from lego.basenet import ResNetLego
    import math

    data_layer = params['data_layer']
    phase = params['phase']
    split = params['split']
    data_dir = params['data_dir']
    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    blocks = params['blocks']
    extra_layer_attach = params['extra_layer_attach']
    extra_blocks = params['extra_blocks']
    extra_num_outputs = params['extra_num_outputs']

    netspec = caffe.NetSpec()

    # data layer
    if data_layer == 'pascal':
        data_params = dict(phase=phase, split=split, data_dir=data_dir)
        data, label = VOCSegDataLego(data_params).attach(netspec)
    else:
        raise Exception('Data layer selected not supported. Available: pascal')

    resnet_params = dict(phase=phase, main_branch=main_branch,
                         num_output_stage1=num_output_stage1,
                         blocks=blocks)
    res_last = ResNetLego(resnet_params).attach(netspec, [data])

    # use_global_stats = False if phase = 'train' else True
    use_global_stats = True
    extrassd_params = dict(base_network='Resnet', main_branch=main_branch,
                           extra_blocks=extra_blocks,
                           extra_num_outputs=extra_num_outputs,
                           use_global_stats=True)
    extra_last = SSDExtraLayersLego(extrassd_params).attach(
        netspec, [netspec[extra_layer_attach]])

    num_classes = params['num_classes']
    min_dim = 300
    mbox_source_layers = params['mbox_source_layers']

    min_ratio = 20
    max_ratio = 95
    step = int(math.floor((max_ratio - min_ratio) /
                          (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    # L2 normalize conv4_3.
    normalizations = [20, 20, 20, -1, -1, -1]
    # normalizations = [-1, -1, -1, -1, -1, -1]

    print min_sizes
    print max_sizes
    print aspect_ratios
    print normalizations
    assemble_params = dict(mbox_source_layers=mbox_source_layers,
                           normalizations=normalizations,
                           aspect_ratios=aspect_ratios,
                           num_classes=num_classes,
                           min_sizes=min_sizes,
                           max_sizes=max_sizes, phase=phase)
    MBoxAssembleLego(assemble_params).attach(netspec, [netspec['label']])

    return netspec
