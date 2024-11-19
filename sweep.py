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
        wandb.log({**trainer.lr, **trainer.metrics}) 
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
                'values': ['auto', 'SGD', 'Adam', 'RMSProp', None]
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
                'min': 0.4
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
    with wandb.init(config=config, job_type="train"):
        config = wandb.config
        model = YOLO("yolo11s.pt")
        add_wandb_callback(model,
                   enable_model_checkpointing=True,
                   )
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        model.train(data="/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml",
                    workers = 0,
                    project="mini-project",
                    name="yolo11s",
                    rect=False,
                    # epochs=config.epochs,
                    epochs=10,
                    
                    augment=config.augment,
                    auto_augment=config.auto_augment,
                    batch=config.batch,
                    box=config.box,
                    close_mosaic=config.close_mosaic,
                    cls=config.cls,
                    copy_paste_mode=config.copy_paste_mode,
                    cos_lr=config.cos_lr,
                    crop_fraction=config.crop_fraction,
                    dfl=config.dfl,
                    erasing=config.erasing, 
                    fliplr=config.fliplr,
                    hsv_h=config.hsv_h,
                    hsv_s=config.hsv_s,
                    hsv_v=config.hsv_v,
                    imgsz=config.imgsz,
                    kobj=config.kobj,
                    lr0=config.lr0,
                    lrf=config.lrf,
                    mask_ratio=config.mask_ratio,
                    momentum=config.momentum,
                    mosaic=config.mosaic,
                    nbs=config.nbs,
                    optimizer=config.optimizer,
                    overlap_mask=config.overlap_mask,
                    patience=config.patience,
                    pose=config.pose,
                    retina_masks=config.retina_masks,
                    scale=config.scale,
                    single_cls=config.single_cls,
                    translate=config.translate,
                    warmup_epochs=config.warmup_epochs,
                    warmup_momentum=config.warmup_momentum,
                    weight_decay=config.weight_decay,
                    )

        
if __name__ == "__main__":
    wandb.agent(sweep_id, function=train)
