import os

from data.dataset import build_dataset
from train.trainer import Trainer
from train.train_utils import (
    get_training_path, 
    get_device,
    get_criterion, 
    get_optimizer, 
    get_activation
)
from model.build_model import build_seg_model
import configuration

# TODO: strcit data input output --> dtype and shape
def pytorch_model_train(cfg):
    exp_path = get_training_path('../checkpoints')
    checkpoint_path = cfg.TRAIN.CHECKPOINT_PATH
    device = get_device()

    # Model
    in_planes = 2*cfg.DATA.SLICE_SHIFT + 1
    model = build_seg_model(
        model_name=cfg.MODEL.NAME, in_planes=in_planes, n_class=cfg.MODEL.N_CLASS)
    
    # Dataset
    train_dataloader, valid_dataloader = build_dataset(
        input_roots, target_roots, train_cases, valid_cases, cfg.DATA.BATCH_SIZE, 
        transform_config=cfg.DATA.TRANSFORM)
    
    # Training
    loss = get_criterion(cfg.TRAIN.LOSS, n_class=cfg.DATA.N_CLASS)
    optimizer = get_optimizer(optimizer_config=cfg.TRAIN.OPTIMIZER, model=model)
    valid_activation = get_activation(cfg.VALID.ACTIVATION)
    trainer = Trainer(model,
                      criterion=loss,
                      optimizer=optimizer,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      logger=logger,
                      device=device,
                      n_class=cfg.DATA.N_CLASS,
                      exp_path=exp_path,
                      train_epoch=cfg.TRAIN.EPOCH,
                      batch_size=cfg.DATA.BATCH_SIZE,
                      valid_activation=valid_activation,
                      history=checkpoint_path,
                      checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS)
    trainer.fit()


def main():
    config_path = 'config/train.yml'
    assert os.path.exists(config_path), 'config file missing'
    train_cfg = configuration.load_config(config_path, dict_as_member=True)
    pytorch_model_train(train_cfg)


if __name__ == '__main__':
    main()