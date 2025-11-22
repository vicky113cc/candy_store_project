# pip install ultralytics
from ultralytics import YOLO
import cv2

# 載入模型
model = YOLO("yolo11_canday.pt")

# 開啟 webcam
cap = cv2.VideoCapture(0)


# 為每個類別分配一個顏色
import random
random.seed(42)
if hasattr(model, 'names'):
    class_names = model.names
    num_classes = len(class_names)
else:
    class_names = {i: str(i) for i in range(80)}
    num_classes = 80

def get_color(cls_id):
    random.seed(cls_id)
    return tuple([int(x) for x in random.choices(range(50, 256), k=3)])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 預測
    results = model(frame)

    # 取得預測結果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = class_names[cls] if cls in class_names else str(cls)
            color = get_color(cls)
            # 畫框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # 顯示名稱與信心值
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    cv2.imshow('YOLOv8 Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()