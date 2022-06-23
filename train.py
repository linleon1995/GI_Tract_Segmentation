from data.dataset import build_dataset
from train.trainer import Trainer

def pytorch_model_train(cfg):
    exp_path = train_utils.create_training_path('checkpoints')
    checkpoint_path = cfg.TRAIN.CHECKPOINT_PATH
    in_planes = 2*cfg.DATA.SLICE_SHIFT + 1
    model = build_model.build_seg_model(model_name=cfg.MODEL.NAME, in_planes=in_planes, n_class=cfg.DATA.N_CLASS, device=get_device())
    
    transform_config = cfg.DATA.TRANSFORM
    train_dataloader, valid_dataloader = dataset(
        input_roots, target_roots, train_cases, valid_cases, cfg.DATA.BATCH_SIZE, 
        transform_config=transform_config)
    
    loss = train_utils.create_criterion(cfg.TRAIN.LOSS, n_class=cfg.DATA.N_CLASS)
    optimizer = train_utils.create_optimizer(lr=cfg.TRAIN.LR, optimizer_config=cfg.TRAIN.OPTIMIZER, model=model)
    valid_activation = train_utils.create_activation(cfg.VALID.ACTIVATION)


    trainer = Trainer(model,
                      criterion=loss,
                      optimizer=optimizer,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      logger=logger,
                      device=get_device(),
                      n_class=cfg.DATA.N_CLASS,
                      exp_path=exp_path,
                      train_epoch=cfg.TRAIN.EPOCH,
                      batch_size=cfg.DATA.BATCH_SIZE,
                      valid_activation=valid_activation,
                      history=checkpoint_path,
                      checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS)
    trainer.fit()


def main():
    pytorch_model_train(train_cfg)


if __name__ == '__main__':
    main()