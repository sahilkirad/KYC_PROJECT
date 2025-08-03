from ultralytics import YOLO
import cv2

# Load model
model_path = 'D:/KYC_Project/models/detection/kyc_model2/weights/best.pt'  # Ensure this path is correct
model = YOLO(model_path)

# Test image
image_path = 'D:/KYC_Project/datasets/processed/test/aadhaar_adhar23.jpg'
img = cv2.imread(image_path)

# Detect with varying confidence
results = model.predict(img, conf=0.3)  # Lowered confidence threshold
detections = results[0]

# Print detections
for box in detections.boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    conf = box.conf.item()
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    print(f"Detected: {class_name} (Confidence: {conf:.2f}) at ({x1}, {y1}, {x2}, {y2})")

# Visualize
annotated_img = results[0].plot()
cv2.imwrite('D:/KYC_Project/outputs/debug/detection_result.jpg', annotated_img)
print("Saved detection result to D:/KYC_Project/outputs/debug/detection_result.jpg")