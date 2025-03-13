import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import tensorflow as tf

# Device Setup: GPU Configuration
devices = tf.config.experimental.list_physical_devices("GPU")
if devices:
    tf.config.experimental.set_memory_growth(devices[0], enable=True)
    print("GPU detected and memory growth enabled.")
else:
    print("No GPU detected.")

# Function to crop the image
def crop_image(image, num_crops, crop_size):
    img_height, img_width, _ = image.shape
    cropped_images = []
    for i in range(num_crops):
        # Randomly select top-left corner for the crop
        x1 = np.random.randint(0, img_width - crop_size)
        y1 = np.random.randint(0, img_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        crop = image[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (256, 256))
        cropped_images.append(crop_resized)
    return np.array(cropped_images)

# List of image paths
image_paths = [
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Bhandarkar_Oriental_Research_Institute-OK_BORI-028_Scan_0533_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Bhandarkar_Oriental_Research_Institute-OK_BORI-028_Scan_0612_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Bharat_ltihas_Samsodhak_Mandal_Pune-ok_BISM-43_Scan_0147_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Bharat_ltihas_Samsodhak_Mandal_Pune-ok_BISM-43_Scan_0149_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Rajasthan_Oriental_Research_Instt_Jodhpur_RORI.Jodhpur_001_Scan_0006_edited.tif',
    r'C:\Users\Keshika\Downloads\Document AI\train\target\Rajasthan_Oriental_Research_Instt_Jodhpur_RORI.Jodhpur_001_Scan_0038_edited.tif'
]

# Variables to control the cropping process
desired_crops_per_image = 167  # To reach approximately 1000 images
crop_size = 256  # Define a fixed crop size (256x256)

cropped_images = []
for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image: {img_path}")
        continue
    image_crops = crop_image(image, desired_crops_per_image, crop_size)
    cropped_images.append(image_crops)

cropped_images = np.concatenate(cropped_images, axis=0)
print(f'Total number of cropped images: {cropped_images.shape[0]}')

cropped_images = cropped_images / 255.0
cropped_images = cropped_images.reshape(-1, cropped_images.shape[1], cropped_images.shape[2], 1)

# Number of epochs set to 50
num_epochs = 10
kernel_size = (7, 7)

# Encoder
input_shape = cropped_images.shape[1:]
encoder_input = Input(shape=input_shape)
x = Conv2D(32, kernel_size, activation='relu', padding='same')(encoder_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPool2D(pool_size=(2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, kernel_size, activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
decoded = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(x)

autoencoder = Model(encoder_input, decoded, name='Denoising_Model')

# Compile the autoencoder model
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

# Save model during training
checkpoint = ModelCheckpoint("denoising_model.keras", save_best_only=True, save_weights_only=False, verbose=1)

# Train the autoencoder model
history = autoencoder.fit(
    cropped_images,  # Input images (cropped)
    cropped_images,  # Target is the same as input for autoencoder
    epochs=num_epochs,
    batch_size=128,
    validation_split=0.25,
    callbacks=[checkpoint],
    verbose=2
)

autoencoder.save('denoising_model_final.keras')

# Plot the training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
