import os
import pandas as pd
import numpy as np
import cv2
import shutil

def extract_identifier(filename):
    # Attempt extraction using both hyphen and underscore separators
    separators = ['-', '_']
    for separator in separators:
        parts = filename.split(separator)
        if len(parts) >= 3:
            identifier = separator.join(parts[:3])
            return identifier
    return None

#%%
# Directories and Files
train_save_dir = r"./tiles-TCGA"
excel_file = r"W:\train_val_cohort.xlsx"
output_dir = r"./tiles-TCGA/Oversampled"

# Load labels from the excel file
df = pd.read_excel(excel_file)
# df["ID"] = df["ID"].replace("-", "_")
study_id_to_label = dict(zip(df["ID"], df["Event"]))

# Load images and their labels
images = []
labels = []
for root, _, files in os.walk(train_save_dir):
    for file in files:
        if file.endswith('.png'):
            study_id = extract_identifier(file)
            if study_id in study_id_to_label:
                labels.append(study_id_to_label[study_id])
                images.append(os.path.join(root, file))

class_counts = {label: labels.count(label) for label in set(labels)}
print(f"Class distribution before oversampling: {class_counts}")

majority_class = 0
minority_class = 1

num_oversample_majority = int(class_counts[majority_class] * 0.20)
num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])

# Define function to flip and save
flip_history = set()  # To store what kind of flips were applied on an image.

def random_flip_save(image_list, output_dir, counter, class_type):
    while True:
        img_path = np.random.choice(image_list, 1)[0]
        
        img = cv2.imread(img_path)
    
        # Check if both flips are already in the history.
        if (f"{img_path}_hflip" in flip_history) and (f"{img_path}_vflip" in flip_history):
            continue  # If both flips exist, continue the loop to try another image.

        flip_type = 'hflip' if np.random.rand() > 0.5 else 'vflip'
        flip_key = f"{img_path}_{flip_type}"
    
        while flip_key in flip_history:
            flip_type = 'hflip' if flip_type == 'vflip' else 'vflip'
            flip_key = f"{img_path}_{flip_type}"

        flip_history.add(flip_key)

        if flip_type == 'hflip':
            img = cv2.flip(img, 1)
        else:
            img = cv2.flip(img, 0)

        flipped_img_name = f"{os.path.basename(img_path).replace('.png', '')}_{flip_type}_{counter}.png"
        output_path = os.path.join(output_dir, flipped_img_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    
        return True  # Image was flipped and saved successfully.

# Oversample majority and minority classes
majority_images = [img for img, label in zip(images, labels) if label == majority_class]
minority_images = [img for img, label in zip(images, labels) if label == minority_class]

oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)

counter = 0
for _ in oversampled_majority:
    output_folder = os.path.join(output_dir, str(majority_class))
    random_flip_save(majority_images, output_folder, counter, majority_class)
    counter += 1

for _ in oversampled_minority:
    output_folder = os.path.join(output_dir, str(minority_class))
    random_flip_save(minority_images, output_folder, counter, minority_class)
    counter += 1

# Save original images
for img_path in images:
    output_path = img_path.replace(train_save_dir, output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(img_path, output_path)

# Print class distribution after oversampling
class_counts[majority_class] += num_oversample_majority
class_counts[minority_class] += num_oversample_minority
print(f"Class distribution after oversampling: {class_counts}")
