# 🧠 *Intel Image Classification with CNN*

## 📌 Project Overview

This project focuses on building a Convolutional Neural Network (CNN) for image classification using the Intel Image Classification Dataset.

The goal is to gain practical experience in deep learning, covering:
* Data preprocessing & augmentation
* CNN model design and training
* Model evaluation with metrics and visualizations
* Interpretation of model decisions using Grad-CAM

## 📂 Dataset
* Source: Intel Image Classification Dataset on Kaggle
* Classes:
  * 🏢 Buildings
  * 🌳 Forest
  * 🧊 Glacier
  * ⛰️ Mountain
  * 🌊 Sea
  * 🛣️ Street
* Size: 25,000+ images (150x150 pixels)

## ⚙️ *Methods & Workflow*
*1. Data Preprocessing*
    * Resizing & normalization
    * Train/validation/test split
    * Data augmentation (rotation, flip, zoom, color jitter, etc.)

*2. Model Development*
    * CNN architecture with:
     * Convolutional layers
     * Pooling layers
     * Dense (fully connected) layers
     * Dropout for regularization
     * Activation functions (ReLU, Softmax)
     * (Optional) Transfer Learning with pretrained models

*3. Model Evaluation*
    * Accuracy & loss plots
    * Confusion Matrix & Classification Report
    * Grad-CAM visualizations for interpretability

*4. Hyperparameter Tuning*
    * Learning rate, batch size, dropout rate, optimizer
    * Experiments with number of layers & filters
    * Overfitting/underfitting analysis

## 📊 Results
Accuracy: ~XX% on test set
Loss: XX
Confusion Matrix & Classification Report: Included in notebook
Grad-CAM: Shows model attention on correct image regions

📌 See notebook for full results and detailed metrics.

🚀 How to Run
> Clone the repository
> git clone https://github.com/your-username/intel-image-classification.git
> cd intel-image-classification

> Open notebook in Kaggle or Jupyter


## Requirements:
* Python 3.x
* TensorFlow / Keras
* NumPy, Matplotlib, Seaborn
* Scikit-learn

## 📎 Links
   📓 Kaggle Notebook
   💻 GitHub Repository

## 📢 Future Improvements

* Add Transfer Learning (ResNet, VGG16, MobileNet)
* Hyperparameter optimization with Keras Tuner
* Experiment tracking with TensorBoard / Weights & Biases

✍️ Prepared by: İzzet Özdemir
