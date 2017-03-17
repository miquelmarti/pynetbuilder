"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.

Modified by Miquel Marti miquelmr@kth.se
"""

from base import BaseLego, BaseLegoFunction

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

from copy import deepcopy

class ConfigDataLego(BaseLego):
    '''
        Generic class to read data layer
        info from config files.
    '''

    def __init__(self, data_file):
        self.data_file = data_file

    def attach(self, netspec):
        return


class ImageDataLego(BaseLego):
    def __init__(self, params):
        if params['include'] == 'test':
            params['include'] = dict(phase=caffe.TEST)
        elif params['include'] == 'train':
            params['include'] = dict(phase=caffe.TRAIN)
        params['image_data_param'] = dict(source=params['source'],
                                          batch_size=params['batch_size'])
        if 'mean_file' in params:
            params['transform_param'] = dict(mean_file=params['mean_file'])
        self._required = ['name', 'source', 'batch_size', 'include']
        super(ImageDataLego, self).__init__(params)

    def _init_default_params(self):
        self._default['ntop'] = 2

    def attach(self, netspec):
        param_packet = self._construct_param_packet()
        data_lego, label_lego = L.ImageData(**param_packet)
        netspec['data'] = data_lego
        netspec['label'] = label_lego
        return data_lego, label_lego


class DeployInputLego(BaseLego):
    '''
    Lego for Input layer for deployment networks
    '''

    def __init__(self, params={}):
        if 'size' in params.keys():
            size = params['size']
        else:
            size = 300
        self.name = 'data'
        self.input_param = dict(shape=dict(dim=[1, 3, size, size]))

    def attach(self, netspec):
        data_lego = L.Input(name=self.name, input_param=self.input_param)
        netspec['data'] = data_lego
        return data_lego


class VOCSegDataLego(BaseLego):
    '''
    Lego for using the python data layers for PascalVOC from
    https://github.com/shelhamer/fcn.berkeleyvision.org
    which needs to be in the python path
    '''

    def __init__(self, params):
        self._required = ['phase', 'split', 'data_dir']
        self._check_required_params(params)
        self.pydata_params = dict(split=params['split'],
                                  mean=(104.00699, 116.66877, 122.67892),
                                  seed=1337)
        if params['phase'] == 'train':
            self.phase = dict(phase=caffe.TRAIN)
            self.pydata_params['sbdd_dir'] = params['data_dir']
            self.pylayer = 'SBDDSegDataLayer'
        elif params['phase'] == 'test':
            self.phase = dict(phase=caffe.TEST)
            self.pydata_params['voc_dir'] = params['data_dir']
            self.pylayer = 'VOCSegDataLayer'

    def attach(self, netspec):
        data, label = L.Python(module='voc_layers', layer=self.pylayer,
                               ntop=2, param_str=str(self.pydata_params),
                               include=self.phase)
        netspec['data'] = data
        netspec['label'] = label
        return data, label


class VOCDetDataLego(BaseLego):
    '''
    Lego for using the AnnotatedData layer with the LMDB for PascalVOC
    with detection labels created with the scripts from:
    https://github.com/weiliu89/caffe/tree/ssd
    '''

    # TODO: Move default params to a config file
    _train_transform_param = {
            'mirror': True,
            'mean_value': [104, 117, 123],
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': 300,
                    'width': 300,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'distort_param': {
                    'brightness_prob': 0.5,
                    'brightness_delta': 32,
                    'contrast_prob': 0.5,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.5,
                    'hue_delta': 18,
                    'saturation_prob': 0.5,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0.0,
                    },
            'expand_param': {
                    'prob': 0.5,
                    'max_expand_ratio': 4.0,
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                }
            }
    _test_transform_param = {
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': 300,
                    'width': 300,
                    'interp_mode': [P.Resize.LINEAR],
                    },
            }
    _batch_sampler = [
            {
                    'sampler': {
                            },
                    'max_trials': 1,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.1,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.3,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.5,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.7,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.9,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'max_jaccard_overlap': 1.0,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            ]

    def __init__(self, params):
        self._required = ['phase', 'label_map_file', 'data_param', 'anno_type']
        self._check_required_params(params)
        if params['phase'] == 'train':
            self.ntop = 2
            self.transform_param = self._train_transform_param
            self.annotated_data_param = dict(
                batch_sampler=self._batch_sampler,
                label_map_file=params['label_map_file']
                )
        else:
            self.ntop = 2
            self.transform_param = self._test_transform_param
            self.annotated_data_param = dict(
                batch_sampler=[{}],
                label_map_file=params['label_map_file']
                )

        self.data_param = params['data_param']

        if params['anno_type'] is not None:
            self.annotated_data_param.update(anno_type=params['anno_type'])

    def attach(self, netspec):
        if self.ntop == 2:
            data, label = L.AnnotatedData(
                name="data", annotated_data_param=self.annotated_data_param,
                data_param=self.data_param, ntop=self.ntop,
                transform_param=self.transform_param
                )
            netspec['data'] = data
            netspec['label'] = label
            return data, label

        else:
            data = L.AnnotatedData(
                name="data", annotated_data_param=self.annotated_data_param,
                data_param=self.data_param, ntop=self.ntop,
                transform_param=self.transform_param
                )
            netspec['data'] = data
            return data


class VOCSegDetDataLego(BaseLego):
    '''
    Lego for using the python data layers for PascalVOC from
    https://github.com/miquelmarti/MultiTask4RT that provides annotations for
    both Semantic Segmentation and Object Detection.
    Needs to be in the python path.
    '''

    def __init__(self, params):
        self._required = ['list_file', 'batch_size']
        self._check_required_params(params)
        batch_size = params['batch_size']
        if batch_size > 1:
            self._required.append('resize_dim')
            self._check_required_params(params)
            resize_dim = params['resize_dim']
        else:
            resize_dim = None
        self.pydata_params = deepcopy(params)
        self.pydata_params.update(dict(mean=(104, 117, 123), seed=1337))
        self.pylayer = 'VOCSegDetDataLayer'

    def attach(self, netspec):
        data, seg_label, det_label = L.Python(
            module='segdet_layers', layer=self.pylayer, ntop=3,
            param_str=str(self.pydata_params))

        netspec['data'] = data
        netspec['seg_label'] = seg_label
        netspec['det_label'] = det_label
        return data, seg_label, det_label
