from ultralytics import YOLO

# Load a model
def main():
    model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-obb.yaml").load("yolo11n-obb.pt")  # build from YAML and transfer weights

# Train the model
    results = model.train(data="ultralytics/cfg/datasets/crack.yaml", epochs=100, imgsz=640)

# 关键：用if __name__ == '__main__'包裹执行入口
if __name__ == '__main__':
    # Windows系统多进程需要的支持（可选，但加上更稳妥）
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 调用训练函数
    main()