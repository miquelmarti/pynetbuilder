"""
@author Miquel Marti miquelmr@kth.se

This module contains all the hybrid legos needed for generating
networks for Semantic Segmentation using Fully Convolutional Networks
with upsampling layers and skip connections.
Adapted from: https://github.com/shelhamer/fcn.berkeleyvision.org
"""

from base import BaseLego
from base import BaseLegoFunction
from base import Config

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
from copy import deepcopy


class CropLego(BaseLego):
    def __init__(self, params):
        self._required = ['name', 'top_to']
        self._check_required_params(params)
        self.crop_params = deepcopy(params)

    def attach(self, netspec, bottom):
        raise Exception("CropLego not working yet")
        # from caffe.coord_map import crop
        #
        # layer = crop(bottom[0], self.crop_params['top_to'])
        # netspec[self.crop_params['name']] = layer
        # return layer


class FCNAssembleLego(BaseLego):
    '''
        TODO
    '''

    def __init__(self, params):
        self._required = ['skip_source_layer', 'num_classes']
        self._check_required_params(params)
        self.skip_source_layer = params['skip_source_layer']
        self.num_classes = params['num_classes']

    def attach(self, netspec, bottom):
        # TODO: Attach the legos

        score_params = dict(name='score_top', bias_filler=dict(type='constant',
                            value=1), bias_term=True, kernel_size=1, pad=0,
                            num_output=self.num_classes)
        score_top = BaseLegoFunction('Convolution',
                                     score_params).attach(netspec, bottom)

        upscore_params = dict(name='upscore2', convolution_param=dict(
                               num_output=self.num_classes, kernel_size=4,
                               stride=2, bias_term=False,
                               weight_filler=dict(type='bilinear')))
        upscore2 = BaseLegoFunction('Deconvolution', upscore_params
                                    ).attach(netspec, [score_top])

        score_params = dict(name='score_skip', kernel_size=1, pad=0,
                            num_output=self.num_classes, bias_term=True,
                            bias_filler=dict(type='constant', value=1))
        score_skip = BaseLegoFunction(
            'Convolution', score_params).attach(
            netspec, [netspec[self.skip_source_layer[0]]])

        crop_params = dict(name='upscore2_c',
                           crop_param=dict(axis=2, offset=0))
        upscore2_c = BaseLegoFunction('Crop', crop_params).attach(
            netspec, [upscore2, netspec['score_skip']])

        eltwise_params = dict(name='fuse_skip', operation=P.Eltwise.SUM)
        fuse_skip = BaseLegoFunction('Eltwise', eltwise_params).attach(
            netspec, [score_skip, upscore2_c])

        upscore_params = dict(name='upscore16', convolution_param=dict(
                               num_output=self.num_classes, kernel_size=32,
                               stride=16, bias_term=False,
                               weight_filler=dict(type='bilinear')))
        upscore16 = BaseLegoFunction('Deconvolution', upscore_params
                                     ).attach(netspec, [fuse_skip])

        # HARDCODED FOR RESNET50 -> Change for map_coords.crop that computes it
        # wrapped as CropLego. Not working yet.
        crop_params = dict(name='scores',
                           crop_param=dict(axis=2, offset=14))
        scores = BaseLegoFunction('Crop', crop_params).attach(
            netspec, [upscore16, netspec['data']])

        loss_param = dict(name='loss', loss_param=dict(normalize=False,
                                                       ignore_label=255))
        loss = BaseLegoFunction('SoftmaxWithLoss', loss_param).attach(
            netspec, [scores, netspec['label']])
