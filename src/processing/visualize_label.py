import cv2
import os

def visualize_label(image_path, label_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get image dimensions
    h, w = img.shape[:2]

    # Read label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id, x, y, width, height = map(float, parts)
            # Convert from normalized to pixel coordinates
            x1 = int((x - width / 2) * w)
            y1 = int((y - height / 2) * h)
            x2 = int((x + width / 2) * w)
            y2 = int((y + height / 2) * h)
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class: {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    output_path = 'D:/KYC_Project/outputs/debug/annotated_image.jpg'
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    image_path = 'D:/KYC_Project/datasets/processed/test/aadhaar_adhar23.jpg'
    label_path = 'D:/KYC_Project/datasets/processed/test/aadhaar_adhar23.txt'
    visualize_label(image_path, label_path)