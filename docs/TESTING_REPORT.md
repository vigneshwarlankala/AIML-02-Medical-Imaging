# Application Testing Report

## Test Environment
- Operating system: Windows
- Python virtual environment: project .venv
- Inference device: CPU
- Backend: FastAPI with Uvicorn
- Frontend: Streamlit
- Model: DenseNet121

## Test Results

| Test | Expected result | Observed result |
|---|---|---|
| Python syntax compilation for API | No syntax errors | Passed |
| Python syntax compilation for dashboard | No syntax errors | Passed |
| API startup | Application starts | Passed |
| API root request | HTTP 200 | Passed |
| Health endpoint | Model ready | Passed |
| Model information endpoint | Checkpoint metadata returned | Passed |
| Model loading | Checkpoint loads | Passed |
| Dashboard startup | Browser interface opens | Passed |
| API-dashboard connection | Connected status displayed | Passed |
| X-ray image upload | Image accepted and previewed | Passed |
| Prediction | 14 scores returned | Passed |
| Results display | Scores shown in table and chart | Passed |
| CSV export | CSV file downloads | Passed |

## Recorded API Health Response
- status: ready
- checkpoint_exists: True
- model_loaded: True
- device: cpu

## Recorded Runtime Model Information
- Model: densenet121
- Loaded epoch: 2
- Best AUC reported by that loaded checkpoint: approximately 0.6776

## Prediction Evidence
A user-tested image upload produced a classification results table containing all 14 target labels and a probability distribution chart.

The CSV download was also confirmed successful by the user.

Example scores visible in the test screenshot:
- Infiltration: 19.26%
- Atelectasis: 8.97%
- Effusion: 8.04%
- Nodule: 7.30%
- Mass: 6.24%

These are model-generated sigmoid scores, not calibrated clinical probabilities.

## Outstanding Tests
- Confirm API reloads the latest best checkpoint after training is finished.
- Verify prediction using the final checkpoint.
- Test invalid file formats and corrupted images.
- Verify all 14 CSV rows and column values.
- Run a repeatability test with the same image.
- Complete independent test-set evaluation.

## Test Conclusion
The core chest X-ray application workflow has been demonstrated successfully. The outstanding tests above remain incomplete and must not be reported as passed.
