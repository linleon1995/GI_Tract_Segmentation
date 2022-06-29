import os
import torch
import torch.nn as nn
import torchvision
from model.model_utils import layers
from model import unet_2d
from model import unet3d
# from model import keras_unet3d
# from model import d2_mask_rcnn
# TODO: Encapsulation with varing first layer, last layer



def build():
    pass


def build_seg_model(model_name, in_planes, n_class, pytorch_pretrained=True):
    segmnetation_model = select_model(model_name, in_planes, n_class, pytorch_pretrained)
    return segmnetation_model


def select_model(model_name, in_planes, n_class, pretrained=True):
    if model_name == '2D-FCN':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained, progress=False)
        model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        model.classifier[4] = nn.Conv2d(512, n_class, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == 'DeepLabv3':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained, progress=False)
        model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == '2D-Unet':
        model = unet_2d.UNet_2d_backbone(in_channels=in_planes, out_channels=n_class, basic_module=layers.DoubleConv)
    elif model_name == '3D-Unet':
        # TODO: activation
        # TODO: align n_class
        n_class -= 1
        model = unet3d.UNet3D(n_class=n_class)
        # TODO:
        # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    else:
        raise ValueError(f'Undefined model of {model_name}.')
    return model

