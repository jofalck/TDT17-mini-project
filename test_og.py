# testing on the original data

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
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

run = wandb.init(
  project="TDT17-mini-project",
  job_type="validation on original data",
)
artifact = run.use_artifact('jofalck-ntnu/TDT17-mini-project/run_bkvhjrmf_model:v57', type='model') # epoch 57 weights best model
artifact_dir = artifact.download()

model_path = f"{artifact_dir}/epoch57.pt"  
model = YOLO(model_path)

add_wandb_callback(model,
                   )


test_yaml = '/cluster/home/jofa/tdt17/TDT17-mini-project/data/data_original_test.yaml'

metrics = model.val(
  data=Path(test_yaml),
  imgsz = 1344,
  workers=0,
  
  project="TDT17-mini-project",
  name=f"yolo-test{current_time}",
)


# metrics = {
#     "mAP_50": results.box.map50,  # Mean Average Precision @ IoU=0.50
#     "mAP_50-95": results.box.map,  # Mean Average Precision @ IoU=0.50:0.95
#     "precision": results.box.precision.mean,
#     "recall": results.box.recall.mean
# }

wandb.log(  # no arguments needed, dataset and settings remembered
    {
        "mAP_50-95": metrics.box.map,  # map50-95
        "mAP_50": metrics.box.map50,  # map50
    }
)

wandb.log({"PR Curve": wandb.Image(f"/cluster/home/jofa/tdt17/TDT17-mini-project/TDT17-mini-project/yolo-test{current_time}/PR_curve.png")})
wandb.log({"F1 curve": wandb.Image(f"/cluster/home/jofa/tdt17/TDT17-mini-project/TDT17-mini-project/yolo-test{current_time}/F1_curve.png")})
wandb.log({"Confusion Matrix": wandb.Image(f"/cluster/home/jofa/tdt17/TDT17-mini-project/TDT17-mini-project/yolo-test{current_time}/confusion_matrix.png")})
wandb.log({"Precision Confidence Curve": wandb.Image(f"/cluster/home/jofa/tdt17/TDT17-mini-project/TDT17-mini-project/yolo-test{current_time}/P_curve.png")})
wandb.log({"Recall": wandb.Image(f"/cluster/home/jofa/tdt17/TDT17-mini-project/TDT17-mini-project/yolo-test{current_time}/R_curve.png")})


# model = YOLO('/cluster/home/jofa/tdt17/TDT17-mini-project/mini-project/yolo11s-25-epoch-sweep5/weights/best.pt')
# model from best jofalck-ntnu/TDT17-mini-project/run_bkvhjrmf_model:v57

run.finish()