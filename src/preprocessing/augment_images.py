import cv2
import os
import numpy as np
import random

def augment_image(image, label_path, output_img_path, output_label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = [list(map(float, line.strip().split())) for line in lines if line.strip()]

    # Augmentation techniques
    aug_type = random.choice(['rotate', 'flip', 'brightness'])
    
    if aug_type == 'rotate':
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug_img = cv2.warpAffine(image, M, (w, h))
        new_labels = []
        for label in labels:
            class_id, x, y, w, h = label
            # Convert to pixel coordinates
            x, y, w, h = x * 640, y * 640, w * 640, h * 640
            # Rotate bounding box (simplified, assumes large boxes)
            new_x = (x - 320) * np.cos(np.radians(angle)) + (y - 320) * np.sin(np.radians(angle)) + 320
            new_y = -(x - 320) * np.sin(np.radians(angle)) + (y - 320) * np.cos(np.radians(angle)) + 320
            new_labels.append([class_id, new_x/640, new_y/640, w/640, h/640])
    
    elif aug_type == 'flip':
        aug_img = cv2.flip(image, 1)  # Horizontal flip
        new_labels = []
        for label in labels:
            class_id, x, y, w, h = label
            new_labels.append([class_id, 1-x, y, w, h])
    
    else:  # brightness
        factor = random.uniform(0.5, 1.5)
        aug_img = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        new_labels = labels  # No change to labels
    
    cv2.imwrite(output_img_path, aug_img)
    with open(output_label_path, 'w') as f:
        for label in new_labels:
            f.write(f"{int(label[0])} {' '.join(map(str, label[1:]))}\n")

def augment_dataset(input_dir, output_dir, num_augmentations=1):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.jpg'):
            img_path = os.path.join(input_dir, file)
            label_path = os.path.join(input_dir, file.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(label_path):
                img = cv2.imread(img_path)
                for i in range(num_augmentations):
                    output_img = os.path.join(output_dir, f"aug_{i}_{file}")
                    output_label = os.path.join(output_dir, f"aug_{i}_{file.rsplit('.', 1)[0]}.txt")
                    augment_image(img, label_path, output_img, output_label)

# Augment training set
augment_dataset('D:/KYC_Project/datasets/processed/train', 'D:/KYC_Project/datasets/processed/train_augmented')