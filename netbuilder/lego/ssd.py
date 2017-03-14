"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.

Modified by Miquel Marti miquelmr@kth.se
"""

from base import BaseLego, BaseLegoFunction

from caffe import layers as L
from caffe import params as P
from copy import deepcopy

'''
    This module contains all the hybrid legos needed for
    generating networks for object detection using
    SSD (Single shot multibox detector:
    https://github.com/weiliu89/caffe/tree/ssd
    The code is inspired/refactored from the above repository.
'''


class PredictionLego(BaseLego):
    '''
        Basic structure for location or confidence prediction layers
    '''

    def __init__(self, params):
        self._required = ['name', 'num_output']
        self._check_required_params(params)
        self.conv_params = deepcopy(params)
        self.conv_params['stride'] = 1
        self.conv_params['kernel_size'] = 3
        self.conv_params['pad'] = 1
        self.perm_params = dict(name=params['name'] + '_perm')
        self.flat_params = dict(name=params['name'] + '_flat',
                                flatten_param=dict(axis=1))

    def attach(self, netspec, bottom):
        from hybrid import ConvBNLego, ConvReLULego
        conv = BaseLegoFunction('Convolution',
                                self.conv_params).attach(netspec, bottom)
        perm = BaseLegoFunction('Permute',
                                self.perm_params).attach(netspec, [conv])
        flat = BaseLegoFunction('Flatten',
                                self.flat_params).attach(netspec, [perm])
        return flat


class DeepPredictionLego(BaseLego):
    '''
        This lego attaches multiple conv layers for non-linear
        prediction units.
        First layer is 3x3, followed by multiple 1x1 units.
        num_outputs specifies the number of filters in each layer
    '''

    def __init__(self, params):
        self._required = ['name', 'num_outputs', 'use_global_stats']
        self._check_required_params(params)
        self.name = params['name']
        self.num_outputs = params['num_outputs']
        self.use_global_stats = params['use_global_stats']
        self.perm_params = dict(name=params['name'] + '_perm')
        self.flat_params = dict(name=params['name'] + '_flat',
                                flatten_param=dict(axis=1))

    def attach(self, netspec, bottom):
        from hybrid import ConvBNReLULego, ConvBNLego, ShortcutLego
        # attach 3x3 conv layer
        conv_params = dict(name=self.name + '_3by3',
                           num_output=self.num_outputs[0],
                           kernel_size=3, pad=1, stride=1,
                           use_global_stats=self.use_global_stats)
        if len(self.num_outputs) == 1:
            last = ConvBNLego(conv_params).attach(netspec, bottom)
        else:
            last = ConvBNReLULego(conv_params).attach(netspec, bottom)

        for i in range(1, len(self.num_outputs)):
            params = dict(name=self.name + '_1by1_' + str(i),
                          num_output=self.num_outputs[i],
                          kernel_size=1, pad=0, stride=1,
                          use_global_stats=self.use_global_stats)
            if i == len(self.num_outputs) - 1:
                last = ConvBNLego(params).attach(netspec, [last])
            else:
                # last = ConvBNReLULego(params).attach(netspec, [last])
                params['main_branch'] = '1by1_normal'
                params['shortcut'] = 'identity'
                last = ShortcutLego(params).attach(netspec, [last])

        perm = BaseLegoFunction('Permute',
                                self.perm_params).attach(netspec, [last])
        flat = BaseLegoFunction('Flatten',
                                self.flat_params).attach(netspec, [perm])

        return flat


class MBoxUnitLego(BaseLego):
    '''
        Structure for attaching multibox prediction unit to a layers in a
        network.
        A mbox unit contains:
        1. Location prediction layers
        2. Confidence prediction layers-1
        3. Prior box layers

        This corresponds to CreateMultiBoxHead function here:
        https://github.com/weiliu89/caffe/blob/ssd/python/caffe/model_libs.py#L581
    '''

    def __init__(self, params):
        self._required = ['name', 'num_classes', 'num_priors_per_location',
                          'min_size', 'max_size', 'aspect_ratio',
                          'use_global_stats']
        self._check_required_params(params)
        self.params = params

    '''
        bottom array should contain from_layer at idx 0
        and data_layer at idx 1
    '''
    def attach(self, netspec, bottom):

        if self.params['type'] == 'deep':
            # Location prediction layers

            deep_mult = self.params['deep_mult']

            num_outputs = []
            for i in range(self.params['depth']):
                num_outputs.append(self.params['num_priors_per_location'] *
                                   4 * deep_mult)
            num_outputs.append(self.params['num_priors_per_location'] * 4)

            loc_params = dict()
            loc_params['name'] = self.params['name'] + '_mbox_loc'
            loc_params['num_outputs'] = num_outputs
            loc_params['use_global_stats'] = self.params['use_global_stats']
            loc = DeepPredictionLego(loc_params).attach(netspec, [bottom[0]])

            # Confidence prediction layers
            num_outputs = []
            for i in range(self.params['depth']):
                num_outputs.append(self.params['num_priors_per_location'] *
                                   self.params['num_classes'] * deep_mult)
            num_outputs.append(self.params['num_priors_per_location'] *
                               self.params['num_classes'])

            conf_params = dict()
            conf_params['name'] = self.params['name'] + '_mbox_conf'
            conf_params['num_outputs'] = num_outputs
            conf_params['use_global_stats'] = self.params['use_global_stats']
            conf = DeepPredictionLego(conf_params).attach(netspec, [bottom[0]])

        else:
            # Confidence prediction layers
            conf_params = dict()
            conf_params['name'] = self.params['name'] + '_mbox_conf'
            conf_params['num_output'] = self.params['num_classes'] * \
                self.params['num_priors_per_location']
            # conf_params['num_outputs'] = [self.params['num_classes'] *
            #                              self.params['num_priors_per_location']]
            # comment below line to go original way of detection heads
            # conf_params['use_global_stats'] = self.params['use_global_stats']
            conf = PredictionLego(conf_params).attach(netspec, [bottom[0]])

            # Location prediction layers
            loc_params = dict()
            loc_params['name'] = self.params['name'] + '_mbox_loc'
            loc_params['num_output'] = \
                self.params['num_priors_per_location'] * 4
            # loc_params['num_outputs'] = \
            #   [self.params['num_priors_per_location'] * 4]
            # comment below line to go original way of detection heads
            # loc_params['use_global_stats'] = self.params['use_global_stats']
            loc = PredictionLego(loc_params).attach(netspec, [bottom[0]])

        # Priorbox layers
        prior_box_params = dict(min_size=self.params['min_size'],
                                aspect_ratio=self.params['aspect_ratio'],
                                flip=True, clip=False, offset=0.5,
                                variance=[0.1, 0.1, 0.2, 0.2])
        if self.params['max_size']:
            prior_box_params['max_size'] = self.params['max_size']

        prior_params = dict(name=self.params['name'] + '_mbox_priorbox',
                            prior_box_param=prior_box_params)
        prior = BaseLegoFunction('PriorBox',
                                 prior_params).attach(netspec, bottom)

        return [loc, conf, prior]


class MBoxAssembleLego(BaseLego):
    '''
        This lego does the following:
        1. Takes a list of layers
        2. Attaches mbox units
        3. Joins them together and attaches MBox Loss
    '''

    def __init__(self, params):
        self._required = ['mbox_source_layers', 'num_classes',
                          'normalizations', 'aspect_ratios', 'min_sizes',
                          'max_sizes', 'label_map_file', 'name_size_file',
                          'output_directory']
        self.params = params

    def attach(self, netspec, bottom):
        # TODO: Take all default params to config file
        label = bottom[0]
        mbox_source_layers = self.params['mbox_source_layers']
        num_classes = self.params['num_classes']
        normalizations = self.params['normalizations']
        aspect_ratios = self.params['aspect_ratios']
        min_sizes = self.params['min_sizes']
        max_sizes = self.params['max_sizes']
        steps = self.params['steps']
        phase = self.params['phase']
        label_map_file = self.params['label_map_file']
        name_size_file = self.params['name_size_file']
        output_directory = self.params['output_directory']
        num_test_image = self.params['num_test_image']

        use_global_stats = False if phase == 'train' else True

        loc = []
        conf = []
        prior = []

        for i, layer in enumerate(mbox_source_layers):
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(layer)
                norm_layer = BaseLegoFunction(
                    'Normalize', dict(
                        name=norm_name, scale_filler=dict(
                            type="constant", value=normalizations[i]),
                        across_spatial=False, channel_shared=False)
                ).attach(netspec, [netspec[layer]])
                layer_name = norm_name
            else:
                layer_name = layer

            # Estimate number of priors per location given provided parameters.
            aspect_ratio = []
            if len(aspect_ratios) > i:
                aspect_ratio = aspect_ratios[i]
                if type(aspect_ratio) is not list:
                    aspect_ratio = [aspect_ratio]
            if max_sizes and max_sizes[i]:
                num_priors_per_location = 2 + len(aspect_ratio)
            else:
                num_priors_per_location = 1 + len(aspect_ratio)

            num_priors_per_location += len(aspect_ratio)

            params = dict(name=layer_name, num_classes=num_classes,
                          num_priors_per_location=num_priors_per_location,
                          min_size=min_sizes[i], max_size=max_sizes[i],
                          step=steps[i],
                          aspect_ratio=aspect_ratio,
                          use_global_stats=use_global_stats)

            params['deep_mult'] = 4
            params['type'] = 'linear'
            # params['type'] = 'deep'
            # params['depth'] = 3

            arr = MBoxUnitLego(params).attach(
                netspec, [netspec[layer_name], netspec['data']])
            loc.append(arr[0])
            conf.append(arr[1])
            prior.append(arr[2])

            mbox_layers = []
            locs = BaseLegoFunction('Concat',
                                    dict(name='mbox_loc',
                                         axis=1)).attach(netspec, loc)
            mbox_layers.append(locs)
            confs = BaseLegoFunction('Concat',
                                     dict(name='mbox_conf',
                                          axis=1)).attach(netspec, conf)
            mbox_layers.append(confs)
            priors = BaseLegoFunction('Concat',
                                      dict(name='mbox_priorbox',
                                           axis=2)).attach(netspec, prior)
            mbox_layers.append(priors)

        # MultiBoxLoss parameters.
        share_location = True
        background_label_id = 0
        train_on_diff_gt = True
        normalization_mode = P.Loss.VALID
        code_type = P.PriorBox.CENTER_SIZE
        ignore_cross_boundary_bbox = False
        neg_pos_ratio = 3.
        loc_weight = (neg_pos_ratio + 1.) / 4.
        mining_type = P.MultiBoxLoss.MAX_NEGATIVE
        multibox_loss_param = {
            'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
            'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
            'loc_weight': loc_weight,
            'num_classes': num_classes,
            'share_location': share_location,
            'match_type': P.MultiBoxLoss.PER_PREDICTION,
            'overlap_threshold': 0.5,
            'use_prior_for_matching': True,
            'background_label_id': background_label_id,
            'use_difficult_gt': train_on_diff_gt,
            'mining_type': mining_type,
            'neg_pos_ratio': neg_pos_ratio,
            'neg_overlap': 0.5,
            'code_type': code_type,
            'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox
            }
        loss_param = {
            'normalization': normalization_mode,
            }

        if phase == 'train':
            mbox_layers.append(label)
            BaseLegoFunction('MultiBoxLoss',
                             dict(name='mbox_loss',
                                  multibox_loss_param=multibox_loss_param,
                                  loss_param=loss_param,
                                  propagate_down=[True, True, False, False])
                             ).attach(netspec, mbox_layers)
        else:
            if phase == 'test':
                mbox_layers.append(label)
                BaseLegoFunction('MultiBoxLoss',
                                 dict(name='mbox_loss',
                                      multibox_loss_param=multibox_loss_param,
                                      loss_param=loss_param,
                                      propagate_down=[True, True,
                                                      False, False])
                                 ).attach(netspec, mbox_layers)

            # parameters for generating detection output.
            det_out_param = {
                'num_classes': num_classes,
                'share_location': True,
                'background_label_id': 0,
                'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
                'save_output_param': {
                    'output_directory': output_directory,
                    'output_name_prefix': "det_test_",
                    'output_format': "VOC",
                    'label_map_file': label_map_file,
                    'name_size_file': name_size_file,
                    'num_test_image': num_test_image,
                    },
                'keep_top_k': 200,
                'confidence_threshold': 0.01,
                'code_type': P.PriorBox.CENTER_SIZE,
                }

            # parameters for evaluating detection results.
            det_eval_param = {
                'num_classes': num_classes,
                'background_label_id': 0,
                'overlap_threshold': 0.5,
                'evaluate_difficult_gt': False,
                'name_size_file': name_size_file,
                }

            conf_name = "mbox_conf"
            reshape_name = "{}_reshape".format(conf_name)
            netspec[reshape_name] = L.Reshape(
                netspec[conf_name], shape=dict(dim=[0, -1, num_classes]))
            softmax_name = "{}_softmax".format(conf_name)
            netspec[softmax_name] = L.Softmax(netspec[reshape_name], axis=2)
            flatten_name = "{}_flatten".format(conf_name)
            netspec[flatten_name] = L.Flatten(netspec[softmax_name], axis=1)
            mbox_layers[1] = netspec[flatten_name]

            if phase == 'test':
                netspec.detection_out = L.DetectionOutput(
                    *mbox_layers, detection_output_param=det_out_param)
                netspec.detection_eval = L.DetectionEvaluate(
                    netspec.detection_out, label,
                    detection_evaluate_param=det_eval_param)
            else:
                netspec.detection_out = L.DetectionOutput(
                    *mbox_layers, detection_output_param=det_out_param)


class SSDExtraLayersLego(BaseLego):
    def __init__(self, params):
        self._required = ['base_network']
        self._check_required_params(params)
        self.base_network = params['base_network']
        if self.base_network == "ResNet":
            self._required.extend(['extra_blocks', 'extra_num_outputs',
                                   'main_branch', 'use_bn'])
            self._check_required_params(params)
            self.extra_blocks = params['extra_blocks']
            self.extra_num_outputs = params['extra_num_outputs']
            self.main_branch = params['main_branch']
            self.use_bn = params['use_bn']
        else:
            raise NotImplementedError("This type of base network"
                                      "is not supported")

    def attach(self, netspec, bottom):
        from hybrid import ShortcutLego
        from base import BaseLegoFunction

        last = bottom

        if not len(self.extra_blocks) > 0:
            print "No added extra layers for SSD"
            return last

        if self.base_network == "ResNet":

            abc = 'abcdefghijklmnopqrstuvwxyz'
            for stage in range(len(self.extra_blocks)):
                for block in range(self.extra_blocks[stage]):
                    if block == 0:
                        if self.use_bn:
                            shortcut = 'projection'
                        else:
                            shortcut = 'noBN_projection'
                        stride = 2
                    else:
                        shortcut = 'identity'
                        stride = 1

                    name = 'res' + str(stage + 6) + abc[block]
                    curr_num_output = self.extra_num_outputs[stage]

                    params = dict(name=name, num_output=curr_num_output,
                                  shortcut=shortcut,
                                  main_branch=self.main_branch,
                                  stride=stride, filter_mult=None,
                                  use_global_stats=False)
                    if self.use_bn:
                        last = ShortcutLego(params).attach(netspec, last)
                    else:
                        params.update(dict(main_branch='noBN_bottleneck'))
                        last = ShortcutLego(params).attach(netspec, last)
                    last = [last]

            # Add global pooling layer to last added layer.
            pool_param = dict(name='pool_last', pool=P.Pooling.AVE,
                              global_pooling=True)
            last = BaseLegoFunction('Pooling', pool_param).attach(
                netspec, last)
        else:
            raise NotImplementedError("This type of base network"
                                      "is not supported")

        print "Added {} extra stages".format(len(self.extra_blocks))
        return last
