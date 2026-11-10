# AIML-02: AI-Powered Medical Imaging System

## 1. Project Overview

The AI-Powered Medical Imaging System is a deep-learning research project focused on medical image analysis.

The currently operational application uses a DenseNet121 deep-learning model to perform multi-label classification of chest X-ray images. It provides prediction scores for 14 different chest conditions through a FastAPI backend and an interactive Streamlit dashboard.

The project also includes a CT image preprocessing pipeline and an MRI data directory to support future expansion into additional medical imaging modalities.

This project is intended for educational and research purposes and is not a clinically validated diagnostic system.

## 2. Project Objectives

- Develop a deep-learning-based medical image analysis application.
- Implement multi-label chest X-ray classification.
- Use DenseNet121 for automated image feature extraction and classification.
- Develop a FastAPI backend for model inference.
- Build an interactive Streamlit dashboard.
- Display prediction scores and visualizations.
- Support CT medical image preprocessing.
- Establish a foundation for future MRI integration.

## 3. Implemented Features

### 3.1 Chest X-ray Classification

- Upload chest X-ray images in PNG and JPEG formats.
- Image resizing and normalization.
- DenseNet121-based multi-label classification.
- Prediction scores for 14 chest conditions.
- FastAPI inference backend.
- Streamlit interactive dashboard.
- Prediction score visualization.
- CSV export of prediction results.

### 3.2 CT Image Preprocessing

The project includes a CT preprocessing pipeline with the following capabilities:

- NIfTI medical volume loading using NiBabel.
- Image and segmentation-label pairing.
- Memory-efficient slice-by-slice processing.
- CT intensity clipping using the Hounsfield Unit range of -1000 to 400.
- Image normalization to the range [0, 1].
- Image and label resizing to a target volume shape of 96 × 96 × 96.
- PyTorch tensor generation.
- JSON preprocessing report generation.

CT preprocessing is a data preparation component. It does not represent a trained CT segmentation or diagnostic model.

### 3.3 MRI Development

An MRI data directory is included in the project structure. MRI-specific preprocessing, model development, and inference functionality are not yet established as operational.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | DenseNet121 |
| Computer Vision | Torchvision |
| Image Processing | Pillow |
| Medical Image Processing | NiBabel |
| Backend Framework | FastAPI |
| API Server | Uvicorn |
| Frontend Framework | Streamlit |
| Data Processing | NumPy and Pandas |
| Testing | Pytest |
| Development Environment | Python Virtual Environment |

### Verified CPU Configuration

- PyTorch: 2.14.1+cpu
- Torchvision: 0.29.1+cpu

These matching CPU builds were verified to import successfully.

## 5. Project Structure

```text
AIML-02-Medical-Imaging/
|
|-- api/
|   |-- main.py
|
|-- frontend/
|   |-- dashboard.py
|
|-- src/
|   |-- preprocess_ct.py
|
|-- scripts/
|   |-- run_training.py
|
|-- models/
|   |-- classification/
|       |-- densenet121_best.pt
|
|-- data/
|   |-- raw/
|   |   |-- xray/
|   |   |-- ct/
|   |   |-- mri/
|   |
|   |-- processed/
|       |-- ct/
|
|-- docs/
|   |-- MODEL_EVALUATION.md
|   |-- PROJECT_DOCUMENTATION.md
|   |-- SYSTEM_ARCHITECTURE.md
|   |-- TESTING_REPORT.md
|
|-- reports/
|   |-- PROJECT_COMPLETION_REPORT.md
|
|-- tests/
|   |-- test_api_smoke.py
|
|-- README.md
|-- requirements.txt