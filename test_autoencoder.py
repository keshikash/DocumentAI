import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from skimage.metrics import peak_signal_noise_ratio as psnr

# Function to crop image with overlap for smoother blending
def crop_image_with_overlap(image, crop_size=256, overlap=32):
    img_height, img_width = image.shape
    crops = []
    positions = []
    step_size = crop_size - overlap
    for y in range(0, img_height - crop_size + 1, step_size):
        for x in range(0, img_width - crop_size + 1, step_size):
            crop = image[y:y + crop_size, x:x + crop_size]
            crops.append(crop)
            positions.append((y, x))
    return np.array(crops), positions, img_height, img_width, step_size

# Load the model
model = load_model('denoising_model_optuna_final.keras')

# Load a test image in grayscale mode
test_image_path = r'C:\Users\Keshika\Downloads\Document AI\train\original\Rajasthan_Oriental_Research_Instt_Jodhpur_RORI.Jodhpur_001_Scan_0038.tif'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
if test_image is None:
    raise ValueError("Error loading test image.")

# Load ground truth image
ground_path = r'C:\Users\Keshika\Downloads\Document AI\train\target\Rajasthan_Oriental_Research_Instt_Jodhpur_RORI.Jodhpur_001_Scan_0038_edited.tif'
ground_truth = cv2.imread(ground_path, cv2.IMREAD_GRAYSCALE)
if ground_truth is None:
    raise ValueError("Error loading ground truth image.")

# Crop the test image with overlapping regions
crops, positions, img_height, img_width, step_size = crop_image_with_overlap(test_image, crop_size=256, overlap=32)
crops = crops / 255.0  # Normalize
crops = crops.reshape(-1, 256, 256, 1)

# Denoise the images
denoised_crops = model.predict(crops)

# Prepare final denoised image with smooth blending for overlap areas
denoised_image = np.zeros((img_height, img_width), dtype=np.float32)
weight_mask = np.zeros((img_height, img_width), dtype=np.float32)

# Gaussian mask for smoother blending of overlapping areas
gaussian_mask = cv2.getGaussianKernel(256, 64) * cv2.getGaussianKernel(256, 64).T
gaussian_mask = gaussian_mask / gaussian_mask.max()  # Normalize mask to max of 1

# Blend denoised crops into the full image
for (y, x), denoised_crop in zip(positions, denoised_crops):
    denoised_crop = denoised_crop.reshape(256, 256)
    denoised_image[y:y + 256, x:x + 256] += denoised_crop * gaussian_mask
    weight_mask[y:y + 256, x:x + 256] += gaussian_mask

# Avoid division by zero by adding a small epsilon to the weight mask
epsilon = 1e-8
denoised_image /= (weight_mask + epsilon)

# Rescale the final image to the 0-255 range
denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)

# Apply Otsu's binarization to the denoised image
_, otsu_binarized = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Flatten images for evaluation
y_true = (ground_truth.flatten() > 127).astype(np.uint8)  # Normalize ground truth
y_pred = (otsu_binarized.flatten() > 127).astype(np.uint8)  # Normalize predicted

# Function to calculate PRF and PRSN metrics
def calculate_prf_prsn(y_true, y_pred):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Precision, Recall, F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Specificity and Negative Predictive Value
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Handle divide by zero
    
    # Print the metrics
    print(f"Precision (P): {precision:.4f}")
    print(f"Recall (R): {recall:.4f}")
    print(f"F1-Score (F): {f1:.4f}")
    print(f"Specificity (SP): {specificity:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    
    return precision, recall, f1, specificity, npv

# Calculate and display metrics
calculate_prf_prsn(y_true, y_pred)

# Calculate PSNR
psnr_value = psnr(ground_truth, denoised_image, data_range=255)
print(f"PSNR Value: {psnr_value:.2f} dB")

# Display the original test image and Otsu Binarized image side by side
plt.figure(figsize=(20, 10))  # Adjust the figure size for side-by-side comparison

# Display the original test image
plt.subplot(1, 2, 1)
plt.imshow(test_image, cmap='gray')
plt.title("Original Test Image", fontsize=20)
plt.axis("off")  # Hide axis

# Display the Otsu binarized image
plt.subplot(1, 2, 2)
plt.imshow(otsu_binarized, cmap='gray')
plt.title("Otsu Binarized Image", fontsize=20)
plt.axis("off")  # Hide axis

plt.show()
