import cv2
import easyocr
from ultralytics import YOLO
import re
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    filename='D:/KYC_Project/logs/ocr_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize EasyOCR reader with Hindi and English languages
reader = easyocr.Reader(['hi', 'en'], gpu=False)  # Set gpu=True if you have CUDA

def validate_aadhaar(text):
    """Validate if text is a 12-digit Aadhaar number."""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', text)
    # Check if it's exactly 12 digits
    if len(digits) == 12:
        return True, digits
    return False, digits

def validate_pan(text):
    """Validate if text is a 10-character PAN number (e.g., ABCDE1234F)."""
    # Remove spaces and convert to uppercase
    text = text.replace(' ', '').upper()
    pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
    if re.match(pattern, text):
        return True, text
    return False, text

def validate_passport(text):
    """Validate if text looks like a passport number."""
    # Remove spaces and convert to uppercase
    text = text.replace(' ', '').upper()
    # Indian passport format: A1234567 or similar
    pattern = r'^[A-Z]\d{7}$'
    if re.match(pattern, text):
        return True, text
    return False, text

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
    """Enhance image for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply light Gaussian blur to reduce noise
    filtered = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def extract_text_with_easyocr(image_path, class_name):
    """Extract text using EasyOCR with specific processing for each document type."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Deskew the image
        img = deskew_image(img)
        
        # Enhance the image
        enhanced = enhance_image(img)
        
        # Save enhanced image for debugging
        debug_dir = 'D:/KYC_Project/outputs/debug'
        os.makedirs(debug_dir, exist_ok=True)
        enhanced_path = os.path.join(debug_dir, f"enhanced_{class_name}_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_path, enhanced)
        
        # Extract text using EasyOCR
        results = reader.readtext(enhanced)
        
        # Extract all text
        all_text = []
        for (bbox, text, prob) in results:
            if prob > 0.3:  # Confidence threshold
                all_text.append(text.strip())
        
        raw_text = ' '.join(all_text)
        logging.info(f"EasyOCR extracted text for {class_name}: {raw_text}")
        
        # Process based on document type
        if class_name == 'aadhaar':
            # Look for 12-digit numbers (Aadhaar format)
            aadhaar_patterns = [
                r'\b\d{4}\s?\d{4}\s?\d{4}\b',  # Standard format with optional spaces
                r'\b\d{12}\b',  # 12 consecutive digits
            ]
            
            for pattern in aadhaar_patterns:
                matches = re.findall(pattern, raw_text)
                if matches:
                    # Clean the matched text
                    cleaned = re.sub(r'\s', '', matches[0])
                    if len(cleaned) == 12:
                        return cleaned, raw_text
            
            # Fallback: extract all digits and take first 12
            digits = re.sub(r'\D', '', raw_text)
            if len(digits) >= 12:
                return digits[:12], raw_text
            else:
                return digits, raw_text
                
        elif class_name == 'pan_card':
            # Look for PAN format: ABCDE1234F
            pan_patterns = [
                r'[A-Z]{5}\d{4}[A-Z]',  # Standard PAN format
                r'[A-Z]{5}\s?\d{4}\s?[A-Z]',  # With optional spaces
            ]
            
            for pattern in pan_patterns:
                matches = re.findall(pattern, raw_text.upper())
                if matches:
                    return re.sub(r'\s', '', matches[0]), raw_text
            
            # Fallback: extract alphanumeric text
            alphanumeric = re.sub(r'[^A-Za-z0-9]', '', raw_text)
            return alphanumeric, raw_text
            
        elif class_name == 'passport':
            # Look for passport number format
            passport_patterns = [
                r'[A-Z]\d{7}',  # Standard format
                r'[A-Z]\s?\d{7}',  # With optional space
            ]
            
            for pattern in passport_patterns:
                matches = re.findall(pattern, raw_text.upper())
                if matches:
                    return re.sub(r'\s', '', matches[0]), raw_text
            
            # Fallback: extract alphanumeric text
            alphanumeric = re.sub(r'[^A-Za-z0-9]', '', raw_text)
            return alphanumeric, raw_text
        
        return raw_text, raw_text
        
    except Exception as e:
        logging.error(f"Error in EasyOCR extraction: {e}")
        return "", ""

def extract_text_from_image(image_path, model_path):
    """Main function to extract text from images using YOLO detection and EasyOCR."""
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

            # Save cropped image for debugging
            debug_dir = 'D:/KYC_Project/outputs/debug'
            os.makedirs(debug_dir, exist_ok=True)
            cropped_path = os.path.join(debug_dir, f"{class_name}_{os.path.basename(image_path)}")
            cv2.imwrite(cropped_path, cropped)

            # Extract text using EasyOCR
            extracted_text, raw_text = extract_text_with_easyocr(cropped_path, class_name)
            
            # Store results
            extracted_data['class'] = class_name
            extracted_data['text'] = extracted_text
            extracted_data['raw_text'] = raw_text

            # Validate based on document type
            if class_name == 'aadhaar':
                is_valid, cleaned_text = validate_aadhaar(extracted_text)
                extracted_data['valid'] = is_valid
                if is_valid:
                    extracted_data['text'] = cleaned_text
            elif class_name == 'pan_card':
                is_valid, cleaned_text = validate_pan(extracted_text)
                extracted_data['valid'] = is_valid
                if is_valid:
                    extracted_data['text'] = cleaned_text
            elif class_name == 'passport':
                is_valid, cleaned_text = validate_passport(extracted_text)
                extracted_data['valid'] = is_valid
                if is_valid:
                    extracted_data['text'] = cleaned_text

            logging.info(f"Final result for {class_name}: {extracted_data}")

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
        print(f"Raw OCR: {result['raw_text']}")
        logging.info(f"Result: {result}") 