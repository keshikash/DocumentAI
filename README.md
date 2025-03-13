# Denoising Historical Manuscripts

## Overview
This project focuses on restoring historical manuscripts using various denoising techniques. The goal is to enhance the readability and preservation of degraded texts by applying image processing and deep learning methods.

## Approaches Used
### 1. Convolution-based AutoEncoders
AutoEncoders are trained to learn the underlying structure of clean manuscript images and reconstruct them from noisy inputs.

### 2. Otsu Binarization
Otsu's method is applied to separate the foreground (text) from the background by dynamically selecting an optimal threshold.

### 3. Median Filtering
This approach helps remove impulse noise (salt-and-pepper noise) by replacing each pixel with the median of its neighborhood.

### 4. Morphological Filtering
Morphological operations such as erosion and dilation are used to refine text structures and remove unwanted noise.

### 5. Gaussian Filtering
A Gaussian blur is applied to smooth the image while preserving important features, reducing high-frequency noise.

## Dataset
The dataset consists of historical manuscript images obtained from the **Indira Gandhi National Centre for the Arts (IGNCA)**. Preprocessing steps include grayscale conversion, resizing, and normalization.

## Implementation
The project is implemented using:
- **Python** with **OpenCV, NumPy, Matplotlib** for image processing
- **TensorFlow/Keras** for AutoEncoder development
- **Scikit-Image** for binarization and filtering techniques

## Results
Evaluation metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are used to compare the effectiveness of each approach.

| Image | Precision | Recall  | F1 Score | PSNR  |
|-------|----------|---------|---------|------|
| 1     | 0.9525   | 0.9181  | 0.9350  | 10 db  |
| 2     | **0.9625**   | 0.9691  | 0.9658  | 12.85 db  |
| 3     | 0.9474   | **0.9871**  | **0.9668**  | 12.88 db  |

## Future Work
- Fine-tuning the AutoEncoder with additional datasets
- Exploring advanced deep learning architectures such as GANs for restoration
- Developing an interactive UI for users to upload and restore their manuscripts

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, contact **your.email@example.com** or open an issue in the repository.

---

_This project is part of ongoing research on historical document restoration._

