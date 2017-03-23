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
        self._required = ['skip_source_layer', 'num_classes',
                          'seg_label', 'normalize', 'phase', 'num_test_image']
        self._check_required_params(params)
        self.skip_source_layer = params['skip_source_layer']
        self.num_classes = params['num_classes']
        self.seg_label = params['seg_label']
        self.normalize = params['normalize']
        self.phase = params['phase']
        self.num_test_image = params['num_test_image']

    def attach(self, netspec, bottom):
        # Change default params
        Config.set_default_params('Convolution', 'bias_term',
                                  True)
        score_params = dict(name='score_top', bias_filler=dict(type='constant',
                            value=1), kernel_size=1, pad=0,
                            num_output=self.num_classes,
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)])
        score_top = BaseLegoFunction('Convolution',
                                     score_params).attach(netspec, bottom)

        upscore_params = dict(name='upscore2', convolution_param=dict(
                               num_output=self.num_classes, kernel_size=4,
                               stride=2, bias_term=False,
                               group=self.num_classes,
                               weight_filler=dict(type='bilinear')))
        upscore2 = BaseLegoFunction('Deconvolution', upscore_params
                                    ).attach(netspec, [score_top])

        score_params = dict(name='score_skip', kernel_size=1, pad=0,
                            num_output=self.num_classes, bias_term=True,
                            bias_filler=dict(type='constant', value=1),
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)])
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
                               group=self.num_classes,
                               weight_filler=dict(type='bilinear')))
        upscore16 = BaseLegoFunction('Deconvolution', upscore_params
                                     ).attach(netspec, [fuse_skip])

        # HARDCODED FOR RESNET50 -> Change for map_coords.crop that computes it
        # wrapped as CropLego. Not working yet.
        crop_params = dict(name='scores',
                           crop_param=dict(axis=2, offset=14))
        scores = BaseLegoFunction('Crop', crop_params).attach(
            netspec, [upscore16, netspec['data']])

        if self.phase != 'deploy':
            loss_param = dict(name='fcn_loss',
                              loss_param=dict(normalize=self.normalize,
                                              ignore_label=255)
                              )
            last = BaseLegoFunction('SoftmaxWithLoss', loss_param).attach(
                netspec, [scores, self.seg_label])
            if self.phase == 'test':
                seg_score_params = dict(num_test_image=self.num_test_image)
                last = SegScoreLayer(seg_score_params).attach(netspec,
                                                              [scores,
                                                               self.seg_label,
                                                               last])
        else:
            softmax_param = dict(name='softmax')
            last = BaseLegoFunction('Softmax', softmax_param).attach(
                netspec, [scores])

        # Reset default params
        Config.set_default_params('Convolution', 'bias_term',
                                  False)

        return last


class SegScoreLayer(BaseLego):
    '''
    Lego for using the python layers from
    https://github.com/miquelmarti/Multitask4RT that implement the scoring
    functionality of https://github.com/shelhamer/fcn.berkeleyvision.org
    score.py script.
    Needs to be in the python path.
    '''

    def __init__(self, params):
        self._required = ['num_test_image']
        self._check_required_params(params)
        self.pydata_params = dict(test_iters=params['num_test_image'])
        self.pylayer = 'SegScoreLayer'
        self.module = 'seg_score_layer'

    def attach(self, netspec, bottom):
        python_param = dict(module=self.module, layer=self.pylayer,
                            param_str=str(self.pydata_params))
        netspec.seg_scores = L.Python(bottom[0], bottom[1], bottom[2],
                                      python_param=python_param,  ntop=0)
