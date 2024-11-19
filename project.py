from dotenv import load_dotenv
import os
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=(api_key))

wandb.init(project="mini-project",
        job_type="training",)

# Load a YOLO model
model = YOLO("yolo11s.pt")


add_wandb_callback(model,
                   enable_model_checkpointing=True,
                   )


def on_train_epoch_end(trainer): 
        """ Log metrics for learning rate, and "metrics" (mAP etc. and val losses). Triggered at the end of each fit epoch. """ 
        wandb.log({**trainer.lr, **trainer.metrics}) 
        # Log the parameters of the top-performing model 
        # best_map = trainer.metrics.get('best_map', None) 
        # if best_map: 
        #         wandb.log({'Best mAP': best_map}) 
        #         wandb.log({'Parameters': trainer.model.parameters()})

model.add_callback("on_train_epoch_end", on_train_epoch_end)

# Train and Fine-Tune the Model
results = model.train(data="/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml", 
            epochs=2, 
            project="mini-project", 
            name="yolo11s",
            batch=4,
            workers=0,
            cache=False,
            patience=20,
            conf=0.001,
            overlap_mask=False,
            plots=True,
            )

# wandb.log({'Best mAP': results['best_map'], 
#            'Last mAP': results['last_map'], 
#            'Precision': results['precision'], 
#            'Recall': results['recall']})

# model.val(
#         conf=0.0001,  
#           )

wandb.finish()

    
# Evaluate the model's performance on the validation set
# results = model.val(workers=1,
#                     batch=32,  # Match the batch size used in training
#                     )


# model.train(task="detect", 
#             mode="train", 
#             model="yolov8n.pt", 
#             data="/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml,"
#             epochs=10,
#             time=None,
#             patience=100,
#             batch=4,
#             imgsz=640,
#             save=True,
#             save_period=-1,
#             cache=False,
#             device=None,
#             workers=1,
#             project=None,
#             name=train29,
#             exist_ok=False,
#             pretrained=True,
#             optimizer=auto,
#             verbose=True,
#             seed=0,
#             deterministic=True,
#             single_cls=False
#             rect=False
#             cos_lr=False
#             close_mosaic=10
#             resume=False
#             amp=True
#             fraction=1.0
#             profile=False
#             freeze=None
#             multi_scale=False
#             overlap_mask=True
#             mask_ratio=4
#             dropout=0.0
#             val=True
#             split=val
#             save_json=False
#             save_hybrid=False
#             conf=None
#             iou=0.7
#             max_det=300
#             half=False
#             dnn=False
#             plots=True
#             source=None
#             vid_stride=1
#             stream_buffer=False
#             visualize=False
#             augment=False
#             agnostic_nms=False
#             classes=None
#             retina_masks=False
#             embed=None
#             show=False
#             save_frames=False
#             save_txt=False
#             save_conf=False
#             save_crop=False
#             show_labels=True
#             show_conf=True
#             show_boxes=True
#             line_width=None
#             format=torchscript
#             keras=False
#             optimize=False
#             int8=False
#             dynamic=False
#             simplify=True
#             opset=None
#             workspace=4
#             nms=False
#             lr0=0.01
#             lrf=0.01
#             momentum=0.937
#             weight_decay=0.0005
#             warmup_epochs=3.0
#             warmup_momentum=0.8
#             warmup_bias_lr=0.1
#             box=7.5
#             cls=0.5
#             dfl=1.5
#             pose=12.0
#             kobj=1.0
#             label_smoothing=0.0
#             nbs=64
#             hsv_h=0.015
#             hsv_s=0.7
#             hsv_v=0.4
#             degrees=0.0
#             translate=0.1
#             scale=0.5
#             shear=0.0
#             perspective=0.0
#             flipud=0.0
#             fliplr=0.5
#             bgr=0.0
#             mosaic=1.0
#             mixup=0.0
#             copy_paste=0.0
#             copy_paste_mode=flip
#             auto_augment=randaugment, 
#             erasing=0.4, 
#             crop_fraction=1.0, 
#             cfg=None,
#             tracker=botsort.yaml,
#             save_dir=runs/detect/train29,
#             )