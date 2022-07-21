import os
from os.path import dirname, basename, isfile, join
import glob
import importlib
import torch
import torch.nn as nn
import torchvision
from model.pytorch.layer import layers
from model.pytorch.base import unet_2d
from model.pytorch.base import unet3d
# from model import keras_unet3d
# from model import d2_mask_rcnn
from inspect import getmembers, isfunction, isclass
from inspect import isclass

# TODO: Encapsulation with varing first layer, last layer


import pkgutil
import importlib
from pprint import pprint

# from libs.base import Base


def load_modules():
    """
    Import all classes under the folder 'modules'.
    >>> load_modules()
    """
    for finder, name, _ in pkgutil.iter_modules([join(dirname(__file__), 'pytorch')]):
        try:
            print(finder.path, name)
            importlib.import_module('{}.{}'.format(finder.path, name))
        except ImportError as e:
            print(e)
            # logger.debug(e)

    # return Base.__subclasses__()


def main():

    for cls in load_modules():
        instance = cls()
        instance.run()


# def build_model(model_name):
    # model_list = check_models()

def get_all_classes():
    """Get all classes name under specific module"""
    pass

def get_model_names():
    """Get all models' name"""
    pass


def build_model(model_name):
    # TODO: import clss only defined inside file
    # TODO: To clarify module and sub-module
    """Build up model from provided model name"""
    from pathlib import Path
    modules = glob.glob(join(dirname(__file__), 'pytorch', '**', "*.py"), recursive=True)
    modules = [ f for f in modules if isfile(f) and not f.endswith('__init__.py')]

    cls_mapping = {}
    for module_path in modules:
        relative_path = os.path.relpath(module_path, os.getcwd())
        dir_parts = Path(relative_path).parts
        module_imp_name = '.'.join(dir_parts)[:-3]
        module = importlib.import_module(module_imp_name)
        classes = getmembers(module, isclass)
        for name, cls in classes:
            print(name, cls, cls.__bases__)
            if cls.__module__.startswith(module_imp_name):
                cls_mapping[f'{module_imp_name}.{name}'] = cls
    pprint(cls_mapping)
    print(3)
    

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

