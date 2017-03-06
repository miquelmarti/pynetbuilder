"""
@author Miquel Marti miquelmr@kth.se
"""

import caffe


def get_vgg_fcn(is_train=True):
    from lego.base import BaseLegoFunction
    from imagenet import VGGNet
    netspec = VGGNet().stitch()

    # TODO: Add extra layers
    raise Exception('VGG not available for FCN')


def get_resnet_fcn(params):
    from lego.data import VOCSegDataLego, DeployInputLego
    from lego.fcn import FCNAssembleLego
    from lego.basenet import ResNetLego

    data_layer = params['data_layer']
    phase = params['phase']
    split = params['split']
    data_dir = params['data_dir']
    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    blocks = params['blocks']
    attach_layer = params['attach_layer']
    skip_source_layer = params['skip_source_layer']
    num_classes = params['num_classes']

    netspec = caffe.NetSpec()

    # data layer
    # data layer
    if phase == 'deploy':
        data_params = dict(size=500)
        data = DeployInputLego(data_params).attach(netspec)
        label = None
    elif data_layer == 'pascal':
        data_params = dict(phase=phase, split=split, data_dir=data_dir)
        data, label = VOCSegDataLego(data_params).attach(netspec)
    else:
        raise Exception('Data layer selected not supported. Available: pascal')

    resnet_params = dict(phase=phase, main_branch=main_branch,
                         num_output_stage1=num_output_stage1,
                         blocks=blocks)
    last = ResNetLego(resnet_params).attach(netspec, [data])

    fcn_params = dict(skip_source_layer=skip_source_layer,
                      num_classes=num_classes, phase=phase,
                      seg_label=label, normalize=False)
    FCNAssembleLego(fcn_params).attach(netspec, [netspec[attach_layer]])

    return netspec
