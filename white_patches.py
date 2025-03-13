import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to crop the image in grayscale
def crop_image(image, num_crops, crop_size):
    # Convert to grayscale if not already
    if image.shape[-1] == 3:  # If the image has 3 channels, convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_height, img_width = image.shape
    cropped_images = []
    for _ in range(num_crops):
        x1 = np.random.randint(0, img_width - crop_size)
        y1 = np.random.randint(0, img_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        crop = image[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (256, 256))
        cropped_images.append(crop_resized)
    return np.array(cropped_images)

# Function to check if an image is all-white
def is_all_white(image, threshold=255):
    return np.all(image == threshold)

# List of image paths
image_paths = [
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\original\BISM-43_Scan_0147.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BISM-43_Scan_0147_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BISM-43_Scan_0148_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BISM-43_Scan_0149_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BORI-028_Scan_0533_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BORI-028_Scan_0543_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\BORI-028_Scan_0612_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\RORI.Jodhpur_001_Scan_0006_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\RORI.Jodhpur_001_Scan_0007_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\ignca-dataV3\train\target\RORI.Jodhpur_001_Scan_0038_edited.tif',
]

# Variables for cropping process
desired_crops_per_image = 167
crop_size = 256

# Crop and save images
cropped_images = []
for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image: {img_path}")
        continue
    image_crops = crop_image(image, desired_crops_per_image, crop_size)
    
    # Filter out all-white crops
    filtered_crops = [crop for crop in image_crops if not is_all_white(crop)]
    cropped_images.extend(filtered_crops)

# Convert to numpy array
cropped_images = np.array(cropped_images)
cropped_images = cropped_images / 255.0  # Normalize to [0, 1]
cropped_images = cropped_images.reshape(-1, 256, 256, 1)  # Reshape for grayscale channel
np.save("cropped_images_filtered.npy", cropped_images)  # Save for training

# Function to display a subset of cropped images
def show_cropped_images(images, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(256, 256), cmap='gray')
        plt.axis('off')
    plt.show()

# Display the first 5 cropped images as a sample
show_cropped_images(cropped_images, num_images=5)
print(f"Total cropped images after filtering: {cropped_images.shape[0]}")
