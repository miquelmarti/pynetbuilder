"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.

Modified by Miquel Marti miquelmr@kth.se
"""

from base import BaseLego

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe

'''
    Generic class to read data layer
    info from config files.
'''
class ConfigDataLego(BaseLego):
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
        params['image_data_param'] = dict(source=params['source'] ,
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
