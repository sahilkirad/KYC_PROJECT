import cv2
import pytesseract
from ultralytics import YOLO
import re
import os
import logging
import numpy as np

# Set Tesseract path (adjust if needed based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set up logging
logging.basicConfig(
    filename='D:/KYC_Project/logs/ocr_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_aadhaar(text):
    """Validate if text is a 12-digit Aadhaar number."""
    pattern = r'^\d{12}$'
    return bool(re.match(pattern, text))

def validate_pan(text):
    """Validate if text is a 10-character PAN number (e.g., ABCDE1234F)."""
    pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
    return bool(re.match(pattern, text))

def deskew_image(image):
    """Correct rotation of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta * 180 / np.pi) - 90
            break
    else:
        angle = 0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def enhance_image(image):
    """Enhance image for better OCR accuracy with Tesseract."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply light Gaussian blur to reduce noise
    filtered = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def extract_text_from_image(image_path, model_path):
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)
        logging.info(f"Loaded YOLOv8 model from {model_path}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Detect objects
        results = model.predict(img, conf=0.3)
        detections = results[0]

        extracted_data = {'class': None, 'text': None, 'valid': False, 'raw_text': None}
        
        for box in detections.boxes:
            # Get class and coordinates
            class_id = int(box.cls)
            class_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the detected region
            cropped = img[y1:y2, x1:x2]

            # Deskew the image
            cropped = deskew_image(cropped)

            # Adjust cropped region for Aadhaar
            if class_name == 'aadhaar':
                h, w = cropped.shape[:2]
                number_region = cropped[int(h*0.05):int(h*0.95), :]  # Middle 90%
                cropped = number_region
                logging.info(f"Cropped region dimensions: {cropped.shape}")

            # Save cropped image for debugging
            debug_dir = 'D:/KYC_Project/outputs/debug'
            os.makedirs(debug_dir, exist_ok=True)
            cropped_path = os.path.join(debug_dir, f"{class_name}_{os.path.basename(image_path)}")
            cv2.imwrite(cropped_path, cropped)

            # Enhance the cropped image
            enhanced = enhance_image(cropped)
            enhanced_path = os.path.join(debug_dir, f"enhanced_{class_name}_{os.path.basename(image_path)}")
            cv2.imwrite(enhanced_path, enhanced)

            # Run Tesseract OCR
            if class_name == 'aadhaar':
                # Use PSM 7 (single line) and restrict to digits for Aadhaar
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            else:
                # Use PSM 6 (block of text) for PAN
                custom_config = r'--oem 3 --psm 6'

            raw_text = pytesseract.image_to_string(enhanced, config=custom_config).strip()
            logging.info(f"Extracted raw text for {class_name}: {raw_text}")

            # Store raw text for debugging
            extracted_data['raw_text'] = raw_text

            # Post-process based on class
            if class_name == 'aadhaar':
                # Look for a 12-digit number in the raw text (with or without spaces)
                aadhaar_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', raw_text)
                if aadhaar_match:
                    extracted_data['text'] = re.sub(r'\s', '', aadhaar_match.group())
                else:
                    # Fallback: Extract all digits and take first 12
                    digits = re.sub(r'\D', '', raw_text)
                    if len(digits) >= 12:
                        extracted_data['text'] = digits[:12]
                    else:
                        extracted_data['text'] = digits
            elif class_name == 'pan_card':
                # Look for a 10-character PAN number
                pan_match = re.search(r'[A-Z]{5}\d{4}[A-Z]', raw_text)
                if pan_match:
                    extracted_data['text'] = pan_match.group()
                else:
                    extracted_data['text'] = raw_text.replace(' ', '')

            # Validate
            extracted_data['class'] = class_name
            if class_name == 'aadhaar':
                extracted_data['valid'] = validate_aadhaar(extracted_data['text'])
            elif class_name == 'pan_card':
                extracted_data['valid'] = validate_pan(extracted_data['text'])

        return extracted_data

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    model_path = 'D:/KYC_Project/models/detection/kyc_model2/weights/best.pt'
    test_image = 'D:/KYC_Project/datasets/processed/test/aadhaar_adhar34.jpg'
    result = extract_text_from_image(test_image, model_path)
    if result:
        print(f"Class: {result['class']}, Text: {result['text']}, Valid: {result['valid']}")
        logging.info(f"Result: {result}")