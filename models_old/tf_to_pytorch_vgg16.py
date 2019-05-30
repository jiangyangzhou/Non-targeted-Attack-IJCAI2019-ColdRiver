import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.vgg_16_conv1_conv1_1_Conv2D = self.__conv(2, name='vgg_16/conv1/conv1_1/Conv2D', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv1_conv1_2_Conv2D = self.__conv(2, name='vgg_16/conv1/conv1_2/Conv2D', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv2_conv2_1_Conv2D = self.__conv(2, name='vgg_16/conv2/conv2_1/Conv2D', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv2_conv2_2_Conv2D = self.__conv(2, name='vgg_16/conv2/conv2_2/Conv2D', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv3_conv3_1_Conv2D = self.__conv(2, name='vgg_16/conv3/conv3_1/Conv2D', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv3_conv3_2_Conv2D = self.__conv(2, name='vgg_16/conv3/conv3_2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv3_conv3_3_Conv2D = self.__conv(2, name='vgg_16/conv3/conv3_3/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv4_conv4_1_Conv2D = self.__conv(2, name='vgg_16/conv4/conv4_1/Conv2D', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv4_conv4_2_Conv2D = self.__conv(2, name='vgg_16/conv4/conv4_2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv4_conv4_3_Conv2D = self.__conv(2, name='vgg_16/conv4/conv4_3/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv5_conv5_1_Conv2D = self.__conv(2, name='vgg_16/conv5/conv5_1/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv5_conv5_2_Conv2D = self.__conv(2, name='vgg_16/conv5/conv5_2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_conv5_conv5_3_Conv2D = self.__conv(2, name='vgg_16/conv5/conv5_3/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_fc6_Conv2D = self.__conv(2, name='vgg_16/fc6/Conv2D', in_channels=512, out_channels=4096, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_fc7_Conv2D = self.__conv(2, name='vgg_16/fc7/Conv2D', in_channels=4096, out_channels=4096, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.vgg_16_fc8_Conv2D = self.__conv(2, name='vgg_16/fc8/Conv2D', in_channels=4096, out_channels=110, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x,get_feature=False):
        vgg_16_conv1_conv1_1_Conv2D_pad = F.pad(x, (1, 1, 1, 1))
        vgg_16_conv1_conv1_1_Conv2D = self.vgg_16_conv1_conv1_1_Conv2D(vgg_16_conv1_conv1_1_Conv2D_pad)
        vgg_16_conv1_conv1_1_Relu = F.relu(vgg_16_conv1_conv1_1_Conv2D)
        vgg_16_conv1_conv1_2_Conv2D_pad = F.pad(vgg_16_conv1_conv1_1_Relu, (1, 1, 1, 1))
        vgg_16_conv1_conv1_2_Conv2D = self.vgg_16_conv1_conv1_2_Conv2D(vgg_16_conv1_conv1_2_Conv2D_pad)
        vgg_16_conv1_conv1_2_Relu = F.relu(vgg_16_conv1_conv1_2_Conv2D)
        vgg_16_pool1_MaxPool = F.max_pool2d(vgg_16_conv1_conv1_2_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vgg_16_conv2_conv2_1_Conv2D_pad = F.pad(vgg_16_pool1_MaxPool, (1, 1, 1, 1))
        vgg_16_conv2_conv2_1_Conv2D = self.vgg_16_conv2_conv2_1_Conv2D(vgg_16_conv2_conv2_1_Conv2D_pad)
        vgg_16_conv2_conv2_1_Relu = F.relu(vgg_16_conv2_conv2_1_Conv2D)
        vgg_16_conv2_conv2_2_Conv2D_pad = F.pad(vgg_16_conv2_conv2_1_Relu, (1, 1, 1, 1))
        vgg_16_conv2_conv2_2_Conv2D = self.vgg_16_conv2_conv2_2_Conv2D(vgg_16_conv2_conv2_2_Conv2D_pad)
        vgg_16_conv2_conv2_2_Relu = F.relu(vgg_16_conv2_conv2_2_Conv2D)
        vgg_16_pool2_MaxPool = F.max_pool2d(vgg_16_conv2_conv2_2_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vgg_16_conv3_conv3_1_Conv2D_pad = F.pad(vgg_16_pool2_MaxPool, (1, 1, 1, 1))
        vgg_16_conv3_conv3_1_Conv2D = self.vgg_16_conv3_conv3_1_Conv2D(vgg_16_conv3_conv3_1_Conv2D_pad)
        vgg_16_conv3_conv3_1_Relu = F.relu(vgg_16_conv3_conv3_1_Conv2D)
        vgg_16_conv3_conv3_2_Conv2D_pad = F.pad(vgg_16_conv3_conv3_1_Relu, (1, 1, 1, 1))
        vgg_16_conv3_conv3_2_Conv2D = self.vgg_16_conv3_conv3_2_Conv2D(vgg_16_conv3_conv3_2_Conv2D_pad)
        vgg_16_conv3_conv3_2_Relu = F.relu(vgg_16_conv3_conv3_2_Conv2D)
        vgg_16_conv3_conv3_3_Conv2D_pad = F.pad(vgg_16_conv3_conv3_2_Relu, (1, 1, 1, 1))
        vgg_16_conv3_conv3_3_Conv2D = self.vgg_16_conv3_conv3_3_Conv2D(vgg_16_conv3_conv3_3_Conv2D_pad)
        vgg_16_conv3_conv3_3_Relu = F.relu(vgg_16_conv3_conv3_3_Conv2D)
        vgg_16_pool3_MaxPool = F.max_pool2d(vgg_16_conv3_conv3_3_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vgg_16_conv4_conv4_1_Conv2D_pad = F.pad(vgg_16_pool3_MaxPool, (1, 1, 1, 1))
        vgg_16_conv4_conv4_1_Conv2D = self.vgg_16_conv4_conv4_1_Conv2D(vgg_16_conv4_conv4_1_Conv2D_pad)
        vgg_16_conv4_conv4_1_Relu = F.relu(vgg_16_conv4_conv4_1_Conv2D)
        vgg_16_conv4_conv4_2_Conv2D_pad = F.pad(vgg_16_conv4_conv4_1_Relu, (1, 1, 1, 1))
        vgg_16_conv4_conv4_2_Conv2D = self.vgg_16_conv4_conv4_2_Conv2D(vgg_16_conv4_conv4_2_Conv2D_pad)
        vgg_16_conv4_conv4_2_Relu = F.relu(vgg_16_conv4_conv4_2_Conv2D)
        vgg_16_conv4_conv4_3_Conv2D_pad = F.pad(vgg_16_conv4_conv4_2_Relu, (1, 1, 1, 1))
        vgg_16_conv4_conv4_3_Conv2D = self.vgg_16_conv4_conv4_3_Conv2D(vgg_16_conv4_conv4_3_Conv2D_pad)
        vgg_16_conv4_conv4_3_Relu = F.relu(vgg_16_conv4_conv4_3_Conv2D)
        vgg_16_pool4_MaxPool = F.max_pool2d(vgg_16_conv4_conv4_3_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vgg_16_conv5_conv5_1_Conv2D_pad = F.pad(vgg_16_pool4_MaxPool, (1, 1, 1, 1))
        vgg_16_conv5_conv5_1_Conv2D = self.vgg_16_conv5_conv5_1_Conv2D(vgg_16_conv5_conv5_1_Conv2D_pad)
        vgg_16_conv5_conv5_1_Relu = F.relu(vgg_16_conv5_conv5_1_Conv2D)
        vgg_16_conv5_conv5_2_Conv2D_pad = F.pad(vgg_16_conv5_conv5_1_Relu, (1, 1, 1, 1))
        vgg_16_conv5_conv5_2_Conv2D = self.vgg_16_conv5_conv5_2_Conv2D(vgg_16_conv5_conv5_2_Conv2D_pad)
        vgg_16_conv5_conv5_2_Relu = F.relu(vgg_16_conv5_conv5_2_Conv2D)
        vgg_16_conv5_conv5_3_Conv2D_pad = F.pad(vgg_16_conv5_conv5_2_Relu, (1, 1, 1, 1))
        vgg_16_conv5_conv5_3_Conv2D = self.vgg_16_conv5_conv5_3_Conv2D(vgg_16_conv5_conv5_3_Conv2D_pad)
        vgg_16_conv5_conv5_3_Relu = F.relu(vgg_16_conv5_conv5_3_Conv2D)
        vgg_16_pool5_MaxPool = F.max_pool2d(vgg_16_conv5_conv5_3_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vgg_16_fc6_Conv2D = self.vgg_16_fc6_Conv2D(vgg_16_pool5_MaxPool)
        vgg_16_fc6_Relu = F.relu(vgg_16_fc6_Conv2D)
        vgg_16_fc7_Conv2D = self.vgg_16_fc7_Conv2D(vgg_16_fc6_Relu)
        vgg_16_fc7_Relu = F.relu(vgg_16_fc7_Conv2D)
        vgg_16_fc8_Conv2D = self.vgg_16_fc8_Conv2D(vgg_16_fc7_Relu)
        vgg_16_fc8_squeezed = torch.squeeze(vgg_16_fc8_Conv2D)
        return vgg_16_fc8_squeezed


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

