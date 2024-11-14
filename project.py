from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs with reduced batch size and image size
results = model.train(data="/cluster/home/jofa/tdt17/TDT17-mini-project/data/data.yaml", epochs=3, batch=8, imgsz=320, workers=0)

# Evaluate the model's performance on the validation set
results = model.val(workers=0)

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")
