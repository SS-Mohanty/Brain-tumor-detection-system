# Automated Brain Tumor Classification 🧠

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Web Application:** [brain-tumor-detection-system-ss-ai.streamlit.app](https://brain-tumor-detection-system-ss-ai.streamlit.app)

## Overview
This repository contains an end-to-end deep learning diagnostic tool that classifies brain MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary, or No Tumor**. 

Beyond standard classification, this project integrates **Explainable AI (XAI)** via Grad-CAM (Gradient-weighted Class Activation Mapping). The application provides real-time side-by-side visual diagnostics, highlighting the exact neural regions that influenced the model's prediction, thereby enhancing clinical trust and interpretability.

## Key Features & Engineering Highlights
* **Custom CNN Architecture:** A lightweight, 3-stage Convolutional Neural Network optimized for 150x150 MRI input dimensions, achieving **90.69% classification accuracy**.
* **Advanced Grad-CAM Artifact Resolution:** Addressed the common XAI issue of low-resolution feature map (17x17) "bleeding" onto skull boundaries. This was resolved by implementing a custom post-processing pipeline utilizing **Gaussian smoothing, tissue thresholding, and a radial center-bias mask**.
* **Dynamic Weight Loading:** To bypass GitHub's strict file size limitations for the 55MB `.keras` model, the weights are hosted via GitHub Releases. The Streamlit application dynamically fetches and caches the weights upon first initialization using `urllib.request`.
* **Professional UI/UX:** Built with Streamlit, featuring a side-by-side diagnostic layout comparing the raw MRI scan directly against the Grad-CAM heatmap.

## Dataset
The model was trained on a comprehensive 4-class Brain Tumor MRI dataset sourced from Kaggle. 
kaggle link - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
* **Classes:** Glioma, Meningioma, Pituitary, No Tumor.
* **Preprocessing:** Images resized to 150x150 pixels, normalized, and augmented to prevent overfitting.

## Deployment Architecture
1. **Frontend/Hosting:** Streamlit Community Cloud
2. **Model Storage:** GitHub Releases (v1.0 tag)
3. **Inference Engine:** TensorFlow / Keras
4. **Image Processing:** SciPy, Pillow, NumPy

## Local Installation & Setup
To run this project locally on your machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/SS-Mohanty/brain-tumor-detection-system.git](https://github.com/SS-Mohanty/brain-tumor-detection-system.git)
cd brain-tumor-detection-system
```
2. Install dependencies
Ensure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt
```
3. Run the application
```bash
streamlit run app.py
```
Note: On the first run, the app will automatically download the model weights (~55MB) into your local directory.

Project Structure
```text
├── app.py                                  # Main Streamlit application and Grad-CAM logic
├── requirements.txt                        # Python dependencies
├── CNN_Brain_Tumor_Seminar_7th_sem.ipynb   # Google Colab notebook containing model training
├── LICENSE                                 # MIT License file
└── README.md                               # Project documentation
```
Author
Subhransu Sekhar Mohanty


License
This project is licensed under the MIT License - see the LICENSE file for details.
