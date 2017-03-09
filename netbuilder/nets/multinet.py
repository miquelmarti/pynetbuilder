"""
@author Miquel Marti miquelmr@kth.se
"""

import caffe
from caffe import params as P


def get_resnet_multi(params):
    from lego.data import VOCSegDetDataLego, DeployInputLego
    from lego.ssd import MBoxUnitLego, MBoxAssembleLego, SSDExtraLayersLego
    from lego.fcn import FCNAssembleLego
    from lego.basenet import ResNetLego
    from lego.base import BaseLegoFunction

    import math

    data_layer = params['data_layer']
    phase = params['phase']
    batch_size_per_device = params['batch_size_per_device']
    num_classes = params['num_classes']
    label_map_file = params['label_map_file']
    name_size_file = params['name_size_file']
    source = params['data_dir']
    output_directory = params['test_out_dir']

    main_branch = params['main_branch']
    num_output_stage1 = params['num_output_stage1']
    blocks = params['blocks']
    attach_layer = params['attach_layer']

    skip_source_layer = params['skip_source_layer']

    extra_blocks = params['extra_blocks']
    extra_num_outputs = params['extra_num_outputs']
    mbox_source_layers = params['mbox_source_layers']
    min_dim = params['min_dim']

    tasks = params['tasks']

    netspec = caffe.NetSpec()

    # data layer
    if phase == 'deploy':
        data_params = dict(size=500)
        data = DeployInputLego(data_params).attach(netspec)
        seg_label = None
        det_label = None
    elif data_layer == 'pascal':
        data_params = dict(phase=phase, list_file=source)
        data, seg_label, det_label = VOCSegDetDataLego(
            data_params).attach(netspec)
    else:
        raise Exception('Data layer selected not supported. Available: pascal')

    # resnet base trunk
    resnet_params = dict(phase=phase, main_branch=main_branch,
                         num_output_stage1=num_output_stage1,
                         blocks=blocks)
    last = ResNetLego(resnet_params).attach(netspec, [data])

    if tasks in ['fcn', 'all']:
        # FCN branch
        fcn_params = dict(skip_source_layer=skip_source_layer,
                          num_classes=num_classes, phase=phase,
                          seg_label=seg_label, normalize=True)
        FCNAssembleLego(fcn_params).attach(netspec, [netspec[attach_layer]])
    elif not phase == 'deploy':
        sil_params = dict(name='silence_seg_label', ntop=0)
        BaseLegoFunction('Silence', sil_params).attach(netspec, [seg_label])

    # SSD branch
    if tasks in ['ssd', 'all']:
        use_global_stats = True
        extrassd_params = dict(base_network='Resnet', main_branch=main_branch,
                               extra_blocks=extra_blocks,
                               extra_num_outputs=extra_num_outputs,
                               use_global_stats=use_global_stats)
        extra_last = SSDExtraLayersLego(extrassd_params).attach(
            netspec, [netspec[attach_layer]])

        min_ratio = 20
        max_ratio = 90
        step = int(math.floor((max_ratio - min_ratio) /
                              (len(mbox_source_layers) - 2)))
        min_sizes = []
        max_sizes = []
        for ratio in xrange(min_ratio, max_ratio + 1, step):
            min_sizes.append(min_dim * ratio / 100.)
            max_sizes.append(min_dim * (ratio + step) / 100.)
        min_sizes = [min_dim * 10 / 100.] + min_sizes
        max_sizes = [min_dim * 20 / 100.] + max_sizes
        steps = [8, 16, 32, 64, 100, 300]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        normalizations = [20, -1, -1, -1, -1, -1]

        assemble_params = dict(mbox_source_layers=mbox_source_layers,
                               normalizations=normalizations,
                               aspect_ratios=aspect_ratios,
                               num_classes=num_classes,
                               min_sizes=min_sizes, steps=steps,
                               max_sizes=max_sizes, phase=phase,
                               output_directory=output_directory,
                               label_map_file=label_map_file,
                               name_size_file=name_size_file)
        MBoxAssembleLego(assemble_params).attach(netspec, [det_label])
    elif not phase == 'deploy':
        sil_params = dict(name='silence_det_label', ntop=0)
        BaseLegoFunction('Silence', sil_params).attach(netspec, [det_label])

    return netspec
