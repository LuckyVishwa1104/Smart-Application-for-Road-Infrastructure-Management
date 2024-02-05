from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

results = model.train(data = 'C:/Users/lvish/anaconda3/envs/crack-model/_final_code/data/data_config.yaml', epochs=55, imgsz=448, batch=8, workers=16)
