import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.nn import BatchNorm3d, InstanceNorm3d, GroupNorm
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264']

def _normalization_3d(inputs, norm='bn'):
    if norm == 'bn':
        return BatchNorm3d(inputs)
    elif norm == 'in':
        return InstanceNorm3d(inputs)
    elif norm == 'gn':
        return GroupNorm(max(32, inputs), inputs)

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm='bn'):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', _normalization_3d(num_input_features, norm))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', _normalization_3d(bn_size * growth_rate, norm))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, norm='bn'):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, norm=norm)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm='bn'):
        super(_Transition, self).__init__()
        self.add_module('norm', _normalization_3d(num_input_features, norm))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self,
                 #spatial_size,
                 #sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000, norm='bn'):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)),
                ('norm0', _normalization_3d(num_init_features, norm)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm=norm)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm=norm)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', _normalization_3d(num_features, norm))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        # Linear layer
        self.classifier = torch.nn.Linear(num_features, num_classes)
    '''
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1,1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    '''

##########################################################################################
##########################################################################################

def densenet121(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model