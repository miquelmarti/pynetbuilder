"""
@author Miquel Marti miquelmr@kth.se

This model contains base networks to be attached after a data layer.
Normally used with some other legos following to define the task,
otherwise isjust a feature extractor.
"""

from caffe import params as P
import caffe

from lego.base import BaseLegoFunction, BaseLego, Config


class ResNetLego(BaseLego):
    '''
        Class to attach a residual network trunk to a data layer
    '''

    def __init__(self, params):
        self._required = ['phase', 'main_branch',
                          'num_output_stage1', 'blocks', 'use_global_stats']
        self.phase = params['phase']
        self.main_branch = params['main_branch']
        self.num_output_stage1 = params['num_output_stage1']
        self.blocks = params['blocks']
        if self.phase == "train":
            self.use_global_stats = params['use_global_stats']
        else:
            self.use_global_stats = True

    def attach(self, netspec, bottom):
        from lego.hybrid import ConvBNReLULego, ShortcutLego

        # Change default param for first conv layer
        Config.set_default_params('Convolution', 'bias_term', True)
        params = dict(name='conv1', num_output=64, kernel_size=7,
                      use_global_stats=self.use_global_stats, pad=3, stride=2)
        stage1 = ConvBNReLULego(params).attach(netspec, bottom)

        # Restore default param
        Config.set_default_params('Convolution', 'bias_term', False)
        params = dict(kernel_size=3, stride=2, pool=P.Pooling.MAX,
                      name='pool1')
        pool1 = BaseLegoFunction('Pooling', params).attach(netspec, [stage1])

        num_output = self.num_output_stage1
        abc = 'abcdefghijklmnopqrstuvwxyz'

        last = pool1
        for stage in range(4):

            for block in range(self.blocks[stage]):
                if block == 0:
                    shortcut = 'projection'
                    if stage > 0:
                        stride = 2
                    else:
                        stride = 1
                else:
                    shortcut = 'identity'
                    stride = 1

                # this is for resnet 18 / 34, where the first block of stage
                # 0 does not need projection shortcut
                if block == 0 and stage == 0 and self.main_branch == 'normal':
                    shortcut = 'identity'

                # This is for not downsampling while creating detection
                # network
                # if block == 0 and stage == 1:
                #    stride = 1

                name = 'res' + str(stage + 2) + abc[block]
                curr_num_output = num_output * (2 ** (stage))

                params = dict(name=name, num_output=curr_num_output,
                              shortcut=shortcut, stride=stride,
                              main_branch=self.main_branch,
                              use_global_stats=self.use_global_stats)
                last = ShortcutLego(params).attach(netspec, [last])
        return last
