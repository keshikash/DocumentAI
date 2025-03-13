import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.metrics import precision_score, recall_score

precision_scores = []
recall_scores = []
f1_scores = []
psnr_scores = []

def round_float(x, d=4):
    return round(x, d)

pred_path = 'C:\Users\Keshika\Downloads\Document AI\test - Copy\test_results\output_1.png'
ground_truth_path = 'C:\Users\Keshika\Downloads\Document AI\test - Copy\target\image_1.tif'

images = os.listdir(pred_path)
thresh = 128

for image in tqdm(images):
    pred_path = os.path.join(pred_path, image)
    gt_path = os.path.join(ground_truth_path, image)

    pred_img = cv2.imread(pred_path)
    gt_img = cv2.imread(gt_path)

    _, pred_img = cv2.threshold(pred_img, thresh, 255, cv2.THRESH_BINARY)
    _, gt_img = cv2.threshold(gt_img, thresh, 255, cv2.THRESH_BINARY)

    pred_img = pred_img / 255.
    gt_img = gt_img / 255.

    pred_img_flatten = pred_img.flatten()
    gt_img_flatten = gt_img.flatten()
    
    ps = precision_score(gt_img_flatten, pred_img_flatten, pos_label=0)
    rs = recall_score(gt_img_flatten, pred_img_flatten, pos_label=0)
    f1 = (2 * ps * rs)/(ps + rs)
    
    precision_scores.append(ps)
    recall_scores.append(rs)
    f1_scores.append(f1)
    
    psnr = cv2.PSNR(gt_img, pred_img)
    psnr_scores.append(psnr)
    
    print(f"File: {image}, Precision: {round_float(ps)}, Recall: {round_float(rs)}, F1 Score: {round_float(f1)}, PSNR: {round_float(psnr)}")

avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_psnr = np.mean(psnr_scores)

print(f"\nAverage Precision: {round_float(avg_precision)}")
print(f"Average Recall: {round_float(avg_recall)}")
print(f"Average F1 Score: {round_float(avg_f1)}")
print(f"Average PSNR: {round_float(avg_psnr)}")