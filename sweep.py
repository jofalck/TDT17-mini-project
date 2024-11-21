import os
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=(api_key))


sweep_config = {
        "name": "mini-project",
        "method": "bayes",
}
metric = {
        "name": "metrics/mAP50-95(B)",
        "goal": "minimize",
}

def on_train_epoch_end(trainer): 
        """ Log metrics for learning rate, and "metrics" (mAP etc. and val losses). Triggered at the end of each fit epoch. """ 
        wandb.log({**trainer.metrics}) 
        # **trainer.lr
        # Log the parameters of the top-performing model 
        # best_map = trainer.metrics.get('best_map', None) 
        # if best_map: 
        #         wandb.log({'Best mAP': best_map}) 
        #         wandb.log({'Parameters': trainer.model.parameters()})



parameters_dict = {
        'auto_augment': {
                'values': ['randaugment', 'autoaugment', 'augmix']
        },
        'augment': {
                'values': [True, False]
        },
        'batch': {
                'values': [-1, 4, 8, 16, 32, 64, 128, 256]
        },
        'box': {
                'distribution': 'uniform',
                'min': 3.75,
                'max': 15
        },
        'close_mosaic': {
                'distribution': 'int_uniform',
                'max': 20,
                'min': 0
        },
        'cls': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.25
        },
        'copy_paste_mode': {
                'distribution': 'categorical',
                'values': ['flip', 'mixup']
        },
        'cos_lr': {
                'distribution': 'categorical',
                'values': [True, False]
        },
        'conf': {
                'distribution': 'uniform',
                'max': 0.75,
                'min': 0.1,
        },
        'crop_fraction': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.1
        },
        'dfl': {
                'distribution': 'uniform',
                'max': 3,
                'min': 0.75
        },
        'epochs': {
                'distribution': 'int_uniform',
                'max': 138,
                'min': 3
        },
        'erasing': {
                'distribution': 'uniform',
                'max': 0.8,
                'min': 0.2
        },
        'fliplr': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.25
        },
        'hsv_h': {
                'distribution': 'uniform',
                'max': 0.03,
                'min': 0.0075
        },
        'hsv_s': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.35
        },
        'hsv_v': {
                'distribution': 'uniform',
                'max': 0.8,
                'min': 0.2
        },
        'imgsz': {
                'distribution': 'int_uniform',
                'max': 1280,
                'min': 320
        },
        'kobj': {
                'distribution': 'int_uniform',
                'max': 2,
                'min': 1
        },
        'lr0': {
                'distribution': 'uniform',
                'max': 0.02,
                'min': 0.005
        },
        'lrf': {
                'distribution': 'uniform',
                'max': 0.02,
                'min': 0.005
        },
        'mask_ratio': {
                'distribution': 'int_uniform',
                'max': 8,
                'min': 2
        },
        'momentum': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.2685
        },
        'mosaic': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0
        },
        'nbs': {
                'distribution': 'int_uniform',
                'max': 128,
                'min': 32
        },
        'optimizer': {
                'distribution': 'categorical',
                'values': ['auto', 'SGD', 'Adam', 'RMSProp']
        },
        'overlap_mask': {
                'distribution': 'categorical',
                'values': [True, False]
        },
        'patience': {
                'distribution': 'int_uniform',
                'max': 40,
                'min': 5
        },
        'pose': {
                'distribution': 'int_uniform',
                'max': 24,
                'min': 6
        },
        'retina_masks': {
                'distribution': 'categorical',
                'values': [True, False]
        },
        'scale': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.25
        },
        'single_cls': {
                'distribution': 'categorical',
                'values': [True, False]
        },
        'translate': {
                'distribution': 'uniform',
                'max': 1,
                'min': 0.05
        },
        'warmup_epochs': {
                'distribution': 'int_uniform',
                'max': 6,
                'min': 2
        },
        'warmup_momentum': {
                'distribution': 'uniform',
                'max': 1.0,
                'min': 0.24
        },
        'weight_decay': {
                'distribution': 'uniform',
                'max': 0.001,
                'min': 0.00025
        },

}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="mini-project")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    with wandb.init(config=config, job_type="sweep"):
        config = wandb.config
        model = YOLO("yolo11s.pt")
        add_wandb_callback(model,
                #    enable_model_checkpointing=True,
                   )
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        data_aug = "/cluster/home/jofa/tdt17/TDT17-mini-project/augmented_data/data.yaml"
        data_norm="/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml"
        model.train(data=data_aug,
                    project="mini-project",
                    name="yolo11s-25-epoch-sweep",
                    rect=False,
                    # epochs=config.epochs,
                    epochs=25,
                    cache=False,
                    
                    warmup_epochs=3,
                    single_cls=True,
                    cos_lr=True,
                    retina_masks=False,
                    augment=False,
                    optimizer=config.optimizer,
                    
                #     workers = config.workers,
                    workers = 0,
                    
                    batch=config.batch,
                    conf=config.conf,
                    hsv_h=config.hsv_h,
                    hsv_s=config.hsv_s,
                    hsv_v=config.hsv_v,
                    imgsz=config.imgsz,
                    lr0=config.lr0,
                    momentum=config.momentum,
                    
                #     lrf=config.lrf,
                    
                #     auto_augment=config.auto_augment,
                #     box=config.box,
                #     close_mosaic=config.close_mosaic,
                #     cls=config.cls,
                #     copy_paste_mode=config.copy_paste_mode,
                #     crop_fraction=config.crop_fraction,
                #     dfl=config.dfl,
                #     erasing=config.erasing, 
                #     fliplr=config.fliplr,
                #     kobj=config.kobj,
                #     mask_ratio=config.mask_ratio,
                #     mosaic=config.mosaic,
                #     nbs=config.nbs,
                #     overlap_mask=config.overlap_mask,
                #     patience=config.patience,
                #     pose=config.pose,
                #     scale=config.scale,
                #     translate=config.translate,
                #     warmup_momentum=config.warmup_momentum,
                #     weight_decay=config.weight_decay,
                    )
        del model
        torch.cuda.empty_cache()
        
def check_workers(config=None):
  """Function that checks the number of workers used in the training process and checks different models. 
  Useful to see if different 

  Args:
      config (wand.config, optional): Declared in a .yaml file. Should not be provided directly. 
  """
  with wandb.init(config=config, job_type="train"):
    config = wandb.config
    model = YOLO(config.model)
    add_wandb_callback(model)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    data_aug = "/cluster/home/jofa/tdt17/TDT17-mini-project/augmented_data/data.yaml"
    model.train(
      data=data_aug,
      name=f"{config.model}-check-workers",
      epochs=2,
      batch=16,
      cache=False,
      workers=config.workers,
    )
        
if __name__ == "__main__":
#   wandb.agent(sweep_id, function=train)
  train()
  # check_workers()