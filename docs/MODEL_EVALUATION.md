\# Model Evaluation Report



\## 1. Overview



This document describes the evaluation of the chest X-ray classification model developed for the AIML-02 Medical Imaging project.



The system uses DenseNet121 to perform multi-label classification of chest X-ray images.



\## 2. Model Architecture



| Parameter | Configuration |

|---|---|

| Model | DenseNet121 |

| Framework | PyTorch |

| Classification Type | Multi-label classification |

| Number of Output Classes | 14 |

| Inference Device | CPU |

| Model Checkpoint | densenet121\_best.pt |



\## 3. Model Checkpoint



The selected model checkpoint is:



`models/classification/densenet121\_best.pt`



The checkpoint corresponds to epoch 5.



\## 4. Evaluation Results



| Evaluation Metric | Result |

|---|---:|

| Best Validation Mean AUC | 0.724471 |

| Rounded Validation Mean AUC | 0.7245 |

| Selected Checkpoint Epoch | 5 |

| Number of Output Classes | 14 |



The recorded validation mean AUC is approximately 0.7245.



AUC measures the model's ability to distinguish between positive and negative examples across classification thresholds.



\## 5. Inference Verification



The model was tested using a real chest X-ray image:



`data/raw/xray/images/00000001\_000.png`



The inference API successfully returned HTTP status code 200.



The response included:

\- Model information.

\- Checkpoint epoch.

\- Best validation AUC.

\- Prediction scores for all 14 output classes.



\## 6. Prediction Interpretation



The model generates a separate prediction score for each output class.



These scores represent model-generated estimates and are not confirmed diagnoses.



The predictions should be interpreted only for educational and research purposes.



\## 7. Limitations



\- The recorded validation AUC does not establish clinical reliability.

\- The model has not undergone clinical validation.

\- Performance may vary across datasets and image acquisition conditions.

\- Prediction scores should not replace professional medical assessment.

\- The reported AUC is a validation metric, not a guarantee of real-world diagnostic performance.



\## 8. Conclusion



The DenseNet121 model successfully performs multi-label chest X-ray inference and returns scores for 14 output classes.



The selected checkpoint achieved a recorded best validation mean AUC of approximately 0.7245 at epoch 5.



The model is integrated with the FastAPI backend and Streamlit dashboard and serves as the primary classification component of the AIML-02 project.

