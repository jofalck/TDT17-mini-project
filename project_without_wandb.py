from dotenv import load_dotenv
from ultralytics import YOLO


model = YOLO('/cluster/home/jofa/tdt17/TDT17-mini-project/mini-project/yolo11s-25-epoch-sweep5/weights/best.pt')



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
                      patience=20,
                      workers=1)
