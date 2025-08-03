import os
import shutil
import random

def split_dataset(image_dir, label_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    total = len(images)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)
    for i, image in enumerate(images):
        if i < train_end:
            dest = train_dir
        elif i < val_end:
            dest = val_dir
        else:
            dest = test_dir
        shutil.copy(os.path.join(image_dir, image), os.path.join(dest, image))
        label_file = image.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dest, label_file))
        else:
            print(f"Warning: Label file {label_file} not found for image {image}")

# Paths
processed_images = 'D:/KYC_Project/datasets/processed/images'
processed_labels = 'D:/KYC_Project/datasets/processed/labels'
train_dir = 'D:/KYC_Project/datasets/processed/train'
val_dir = 'D:/KYC_Project/datasets/processed/val'
test_dir = 'D:/KYC_Project/datasets/processed/test'

# Split dataset
split_dataset(processed_images, processed_labels, train_dir, val_dir, test_dir)