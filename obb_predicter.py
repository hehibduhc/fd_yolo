from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("weighs/yolo11n-obb.pt")

# Run inference on an image
results = model("P0075.png",save = True)  # results list

# View results
for r in results:
    print(r.obb)  # print the OBB object containing the oriented detection bounding boxes