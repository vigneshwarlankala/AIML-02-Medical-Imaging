# System Architecture

## 1. System Overview
The Medical Imaging System is a research prototype for multi-label chest X-ray classification using a DenseNet121 deep learning model.

## 2. Architecture

```mermaid
flowchart TD
    A[Chest X-ray Image] --> B[Image Preprocessing]
    B --> C[DenseNet121 Model]
    C --> D[14-Class Predictions]
    D --> E[FastAPI Backend]
    E --> F[Streamlit Dashboard]
    F --> G[Prediction Scores and Chart]
    F --> H[CSV Download]
