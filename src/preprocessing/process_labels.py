import os

def process_labels(label_dir, output_dir, prefix, new_class):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            src = os.path.join(label_dir, file)
            dst = os.path.join(output_dir, prefix + file)
            with open(src, 'r') as f:
                lines = f.readlines()
            with open(dst, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x, y, w, h = map(float, parts[:5])
                        if w > 0.8 and h > 0.8:  # Filter for large bounding boxes
                            f.write(f"{new_class} {x} {y} {w} {h}\n")
            print(f"Processed label: {file} to {dst}")

# Paths
processed_labels = 'D:/KYC_Project/datasets/processed/labels'

# Process Aadhaar labels (set to class 0)
process_labels('D:/KYC_Project/labels', processed_labels, 'aadhaar_', new_class=0)

# Process PAN card labels (set to class 1)
process_labels('D:/KYC_Project/labels-high-res', processed_labels, 'pan_', new_class=1)