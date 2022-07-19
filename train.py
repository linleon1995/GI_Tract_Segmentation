import os

from data.dataset import build_dataset
from train.trainer import Trainer
from train.train_utils import (
    get_training_path, 
    get_device,
    get_criterion, 
    get_optimizer, 
    get_activation,
    config_logging
)
from model.build_model import build_seg_model, build_model
import configuration

# TODO: strcit data input output --> dtype and shape
# TODO: work more on logger and trainer
# TODO: validaion, testing steps using trainer?
# TODO: Think about this, Can this trainer work on tasks: 3d segmentation, 1d regression, 2d object detection
# TODO: fold
# TODO: display step
# TODO: remove ['out'] in DiceLoss
# TODO: U-net error

def pytorch_model_train(cfg):
    exp_path = get_training_path('../checkpoints')
    checkpoint_path = cfg.TRAIN.CHECKPOINT_PATH
    device = get_device()

    # Model
    in_planes = 2*cfg.DATA.SLICE_SHIFT + 1
    model = build_model('pytorch.base.unet_2d')
    # model = build_seg_model(
    #     model_name=cfg.MODEL.NAME, in_planes=in_planes, n_class=cfg.MODEL.N_CLASS)
    
    # Dataset
    train_dataloader, valid_dataloader = build_dataset(
        cfg.DATA.GI.DATA_ROOT, fold=0, num_fold=5, batch_size=cfg.DATA.BATCH_SIZE, num_class=cfg.MODEL.N_CLASS)

    # Training
    loss = get_criterion(cfg.TRAIN.LOSS, n_class=cfg.MODEL.N_CLASS)
    optimizer = get_optimizer(optimizer_config=cfg.TRAIN.OPTIMIZER, model=model)
    valid_activation = get_activation(cfg.VALID.ACTIVATION)
    trainer = Trainer(model,
                      criterion=loss,
                      optimizer=optimizer,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      device=device,
                      n_class=cfg.MODEL.N_CLASS,
                      exp_path=exp_path,
                      train_epoch=cfg.TRAIN.EPOCH,
                      batch_size=cfg.DATA.BATCH_SIZE,
                      valid_activation=valid_activation,
                      history=checkpoint_path,
                      checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS)
    config_logging(os.path.join(exp_path, 'config.txt'), cfg, access_mode='w+')
    trainer.fit()


def main():
    config_path = 'config/train.yml'
    assert os.path.exists(config_path), 'config file missing'
    train_cfg = configuration.load_config(config_path, dict_as_member=True)
    pytorch_model_train(train_cfg)



if __name__ == '__main__':
    main()
    # preprocess_lung()
    # data_frame =
    # table_image(data_frame)
    

    # # import SimpleITK as sitk
    # # import numpy as np
    # # root = rf'C:\Users\test\Desktop\Leon\Projects\ai-assisted-annotation-client\slicer-plugin\NvidiaAIAA'
    # # f = rf'tmpkvmkf1io.nii.gz'
    # # # itk_img = sitk.ReadImage(os.path.join(root, f))
    # # # img = sitk.GetArrayFromImage(itk_img)
    # # # img = np.uint8(img)
    # # # img_b = img.tobytes()
    # # # print(len(img_b))
    # # # with open(os.path.join(root, 'test.nii.gz'), 'wb') as fw:
    # # #     fw.write(img_b)
    # # import nrrd
    # # f = rf'C:\Users\test\Desktop\Leon\Weekly\0621\TMH\fold4\images\2466138720832919377868467891188675\nrrd\2466138720832919377868467891188675.seg.nrrd'
    # # image = sitk.ReadImage(f)
    # # data, header = nrrd.read(f)
    # # data = np.uint8(data)
    # # itk_image = sitk.Image(data)
    # # print(3)

    # import nrrd
    # import numpy as np
    # import SimpleITK as sitk
    # f = rf'C:\Users\test\AppData\Local\NA-MIC\Slicer 4.13.0-2022-04-26\result\38467158349469692405660363178115017.nrrd'
    # # data, header = nrrd.read(f)
    # # print(data.shape)
    # f2 = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\merge\TMH0003\raw\38467158349469692405660363178115017.mhd'
    # data, header = nrrd.read(f)
    # # data = np.transpose(data, (2, 1, 0))
    # # data = np.load(f)
    # print(data.shape)

    # data = sitk.ReadImage(f2)
    # print(sitk.GetArrayFromImage(data).shape)
