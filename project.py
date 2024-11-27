import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=(api_key))

wandb.init(project="TDT17-mini-project",
  job_type="training",)


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Load a YOLO model
# model = YOLO("yolo11s.pt")
# model = YOLO('/cluster/home/jofa/tdt17/TDT17-mini-project/mini-project/yolo11s79/weights/last.pt')
model = YOLO('/cluster/home/jofa/tdt17/TDT17-mini-project/mini-project/yolo11s-25-epoch-sweep5/weights/best.pt')


add_wandb_callback(model,
                   enable_model_checkpointing=True,
                   )


def on_train_epoch_end(trainer): 
  """ Log metrics for learning rate, and "metrics" (mAP etc. and val losses). Triggered at the end of each fit epoch. """ 
  wandb.log({**trainer.lr, **trainer.metrics}) 

model.add_callback("on_train_epoch_end", on_train_epoch_end)

data_aug = "/cluster/home/jofa/tdt17/TDT17-mini-project/augmented_data/data.yaml"

results = model.train(data=data_aug,
                      batch=16,
                      conf=0.31685501046027204,
                      hsv_h=0.01579634466599653,
                      hsv_s=0.8695094684209294,
                      hsv_v=0.7438035826433458,
                      lr0=0.0223007054062638,
                      momentum=0.4160743757243235,
                      optimizer="SGD",
                      epochs=150,
                      patience=50,
                      workers=1,
                      
                      rect=False,
                      augment=False,
                      warmup_epochs=3,
                      single_cls=True,
                      cos_lr=True,
                      retina_masks=False,
                      
                      project="TDT17-mini-project",
                      name=f"yolo-full {current_time}",
                      )


model.val(data=Path('/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml'),
          )

# Train and Fine-Tune the Model
# results = model.train(data=data_aug,
#                       epochs=100,
#                       patience=20,
#                       workers=0,
                      
#                       rect=False,
#                       augment=False,
#                       warmup_epochs=3,
#                       single_cls=True,
#                       cos_lr=True,
#                       retina_masks=False,
                      
#                       batch=16,
#                       imgsz=1440,
#                       optimizer='SGD',
#                       conf=0.3,
#                       hsv_h=0.025,
#                       hsv_s=0.77,
#                       hsv_v=0.86,
#                       lr0=0.006,
#                       momentum=0.61,
                      
#             project="TDT17-mini-project", 
#             name="yolo-full",
#             )

# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category

wandb.finish()