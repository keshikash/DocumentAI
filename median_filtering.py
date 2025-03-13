import numpy as np
from PIL import Image
import os

def median_filter(data, filter_size):
    """Applies a median filter to a 2D NumPy array."""
    indexer = filter_size // 2
    padded_data = np.pad(data, pad_width=indexer, mode='constant', constant_values=0)
    data_final = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Extract the neighborhood
            neighborhood = padded_data[i:i + filter_size, j:j + filter_size]
            # Calculate median
            data_final[i, j] = np.median(neighborhood)
    
    return data_final

def main():
    # Base paths for input and output directories
    input_base_path = r"C:\Users\Keshika\Downloads\Document AI\test - Copy\target"
    output_base_path = r"C:\Users\Keshika\Downloads\Document AI\test - Copy\test_results"
    
    # Ensure the output directory exists
    os.makedirs(output_base_path, exist_ok=True)
    
    # Process images from image_1.tif to image_4.tif
    for img_index in range(1, 5):
        # Construct the full input and output file paths
        input_filename = os.path.join(input_base_path, f"image_{img_index}.tif")
        output_filename = os.path.join(output_base_path, f"output_{img_index}.png")
        
        # Load the image and convert it to grayscale
        img = Image.open(input_filename).convert("L")
        arr = np.array(img)
        
        # Apply median filter with filter size 10
        removed_noise = median_filter(arr, 10)
        
        # Convert the filtered array back to an image and save it
        img_filtered = Image.fromarray(np.uint8(removed_noise))
        img_filtered.save(output_filename)
        print(f"Denoised image saved as {output_filename}")

main()
