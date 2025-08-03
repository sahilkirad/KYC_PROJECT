import cv2
import os

def clean_images(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Removing corrupted file: {file_path}")
            os.remove(file_path)

# Clean Aadhaar images
clean_images('D:/KYC_Project/images')

# Clean PAN card images
clean_images('D:/KYC_Project/images-high-res')