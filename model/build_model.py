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
    # main()
    """Build up model from provided model name"""
    # import pkgutil
    # it = pkgutil.iter_modules([dirname(__file__), join(dirname(__file__), 'pytorch')])
    # it2 = pkgutil.walk_packages([dirname(__file__), join(dirname(__file__), 'pytorch')])
    # for a in it2:
    #     print(a)
    # for finder, name, _ in it:
    #     print(finder.path, name)
    #     importlib.import_module('{}.{}'.format(finder.path, name))

    # print(__file__)
    # a = join(dirname(__file__), 'pytorch', "*.py")
    modules = glob.glob(join(dirname(__file__), 'pytorch', '**', "*.py"), recursive=True)
    modules = [ f for f in modules if isfile(f) and not f.endswith('__init__.py')]
    # __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

    for module in modules:
        relative_path = os.path.relpath(module, os.getcwd())
        from pathlib import Path
        dir_parts = Path(relative_path).parts
        module_path = '.'.join(dir_parts)[:-3]
        module = importlib.import_module(module_path)
        # module = importlib.import_module(f'model.pytorch.base.unet_2d')
        classes = getmembers(module, isclass)
        print(classes)
        for cls in classes:
            print(cls[0], cls[1].__bases__)

    print(3)
    # model = importlib.import_module(f'model.{model_name}')
    # u = model.unet_2d
    # subclass = model.__subclasses__()
    # unet = model.UNet_2d
    # unet2 = unet(input_channels=1, num_class=4)
    # print(model)
    # return model


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

