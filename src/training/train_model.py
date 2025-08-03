from ultralytics import YOLO
import os
import torch
import logging

# Set up logging
logging.basicConfig(
    filename='D:/KYC_Project/logs/train_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # 'n' for nano, use 'yolov8s.pt' for small, etc.
    logging.info("Loaded YOLOv8 model: yolov8n.pt")

    # Train the model
    model.train(
        data='D:/KYC_Project/config/kyc_data.yaml',
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size (matches preprocessing)
        batch=4,  # Batch size (adjust based on hardware)
        name='kyc_model',  # Name for the training run
        project='D:/KYC_Project/models/detection',  # Save location
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    logging.info("Training completed")

    # Evaluate on validation set
    results = model.val()
    print("Validation Results:", results)
    logging.info(f"Validation Results: {results}")

except Exception as e:
    print(f"Error during training: {e}")
    logging.error(f"Error during training: {e}")
    raise