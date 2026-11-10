# Technical Project Documentation

## 1. Introduction
The AIML-02 Medical Imaging System is an educational deep-learning prototype that applies multi-label classification to chest X-ray images.

## 2. Objective
The objective is to build a workflow that accepts a chest X-ray image, preprocesses it, performs inference using DenseNet121, and displays scores for 14 target labels.

## 3. Dataset
The project uses the NIH ChestX-ray14 dataset.

Dataset inspection recorded:
- CSV records: 112,120
- Valid image records found: 52,120
- Referenced image files missing: 60,000
- Training split: 36,484
- Validation split: 7,818
- Test split: 7,818

The split totals 52,120 records. Missing image references are a dataset limitation and should be disclosed in project evaluation.

## 4. Target Labels
The 14 classification labels are:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## 5. Image Preprocessing
The training pipeline:
- Opens images and converts them to RGB.
- Resizes images to 224 x 224 pixels using Lanczos.
- Converts pixel values to floating-point values in the range 0 to 1.
- Applies ImageNet normalization.

Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]

Training augmentation includes random horizontal flipping and gamma correction.

## 6. Model Architecture
The classification model is DenseNet121 with:
- ImageNet-pretrained backbone during training.
- Dropout probability: 0.2.
- Final fully connected layer: 1024 input features and 14 outputs.
- Sigmoid activation during inference.
- Multi-label binary cross-entropy training objective (BCEWithLogitsLoss).

Each output represents a model-generated score for one label.

## 7. Training Configuration
Recorded configuration:
- Image size: 224
- Batch size: 8
- Maximum configured epochs: 30
- Learning rate: 0.0001
- Dropout: 0.2
- Device: CPU
- Data loader workers: 0

Progressive training phases:
- Epochs 1-5: classifier training.
- Epochs 6-15: selected later DenseNet layers and classifier.
- Epochs 16-30: all model layers.

## 8. Backend
The FastAPI application is implemented in api/main.py.

Endpoints:
- GET / : API information.
- GET /health : service and model readiness.
- GET /model-info : checkpoint and model metadata.
- POST /predict : accepts a PNG/JPEG image and returns 14 scores.

The model is loaded during application startup.

## 9. Dashboard
The Streamlit dashboard is implemented in frontend/dashboard.py.

Features:
- API connectivity status.
- Model information display.
- Image upload and preview.
- Prediction request.
- Results table.
- Probability distribution chart.
- CSV download.

## 10. CT and MRI Scope
CT data extraction was completed for the Task06_Lung dataset. A separate CT preprocessing script exists.

MRI Task01_BrainTumour archive extraction was not confirmed complete in the recorded project inspection.

CT and MRI model training, inference endpoints, and dashboard integration are not verified as completed. They are future development tasks.

## 11. Limitations
- The dataset inspection found many missing referenced image files.
- Training was CPU-based and computationally expensive.
- The recorded model metrics are validation results, not clinical performance evidence.
- CT and MRI workflows are not yet integrated into the tested application.
- The prototype has not been clinically validated.

## 12. Conclusion
A working chest X-ray classification prototype has been implemented using DenseNet121, FastAPI, and Streamlit. The tested workflow includes image upload, model inference, display of 14 scores, visualization, and CSV export.
