import cv2
import os

def standardize_images(input_path, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(input_path):
        img_path = os.path.join(input_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (640, 640))
            output_file = os.path.join(output_path, prefix + file.rsplit('.', 1)[0] + '.jpg')
            cv2.imwrite(output_file, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"Processed: {file} to {output_file}")

# Paths
processed_images = 'D:/KYC_Project/datasets/processed/images'

# Standardize Aadhaar images
standardize_images('D:/KYC_Project/images', processed_images, 'aadhaar_')

# Standardize PAN card images
standardize_images('D:/KYC_Project/images-high-res', processed_images, 'pan_')