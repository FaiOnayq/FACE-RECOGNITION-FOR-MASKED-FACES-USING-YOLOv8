# FACE-RECOGNITION-FOR-MASKED-FACES-USING-YOLOv8
A deep learning-based computer vision project that uses YOLOv8 to detect face masks in images and videos. The model identifies three classes: with_mask, without_mask, and mask_weared_incorrect. Built for real-time applications in public safety and lab access control.


## 📘 Introduction

### 🔍 Problem Definition
Wearing face masks has become essential during the **COVID-19 pandemic** and in environments like labs dealing with hazardous materials. Manual monitoring is inefficient and impractical in large public areas, hence the need for an automated solution using AI/ML.

### 💡 Why It's Interesting
- Real-time monitoring and feedback
- Reduces human effort and fatigue
- High accuracy in detecting correct, incorrect, and no mask usage
- Adaptable to various environments


## 📁 Dataset

We used the **Face Mask Dataset** from [Roboflow]([https://roboflow.com](https://universe.roboflow.com/t1-bh7n0/mask_detection_combined-qbivp)), containing 2699 labeled images categorized into:
- `0: mask_weared_incorrect`
- `1: with_mask`
- `2: without_mask`

### 🔍 Dataset Splitting and Balancing
- Stratified splitting into training (80%), validation (10%), and test (10%) sets
- Undersampling was applied to balance class distribution

## 🛠 Methodology

### 🧠 Model Architecture
- **YOLOv8**: Chosen for its efficiency and high performance in real-time object detection.
- Custom-trained on the face mask dataset

### 🔧 Hyperparameter Tuning
- Optimized learning rate, batch size, optimizer (AdamW performed best), dropout, and mixup
- Achieved optimal results with:
  - Learning Rate: 0.01
  - Batch Size: 16
  - Optimizer: AdamW
  - Dropout: 0.2
  - Mixup: 0.5

## 📊 Evaluation Metrics

- **mAP@0.5**: 93.08%
- **Precision**: 87.9%
- **Recall**: 93.4%
- Confusion Matrix, IoU, F1 Score, Precision-Recall curves

## 🖥 Experimental Setup

### Hardware
- GPU: T4 × 2 / Tesla P100 (via Kaggle)

### Software & Tools
- Python 3.x
- Libraries: NumPy, Pandas, OpenCV, Matplotlib, Pillow, iterstrat
- Framework: Ultralytics YOLOv8
- Dataset Source: Roboflow


