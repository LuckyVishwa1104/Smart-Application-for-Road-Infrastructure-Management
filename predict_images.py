from ultralytics import YOLO
import cv2
import math

model = YOLO("./runs/detect/train4/weights/best.pt")
img = cv2.imread("C:/Users/lvish/anaconda3/envs/crack-model/_final_code/data/val/images/c-5.jpg")
results = model(img, stream=True, conf=0.08)
for r in results:
      boxes = r.boxes
      for box in boxes:
          #Boundding Box
          x1,y1,x2,y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((box.conf[0])*100)/100
          cv2.rectangle(img, (x1, y1), (x2, y2), (225, 225, 0), 4)
          cv2.putText(img, f'Confidence: {conf}', (x1+33, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (196, 184, 6), 1)
cv2.imshow("Result",img)
cv2.waitKey(0)
