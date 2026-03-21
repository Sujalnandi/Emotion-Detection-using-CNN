# Real-Time Facial Emotion Detection using Deep Learning

This project is a B.Tech final-year AI/ML system for classifying facial emotions from images and webcam streams.

## Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Improved Training Pipeline

- Strong data augmentation: rotation, shifts, zoom, flip, brightness, shear
- Proper normalization (`rescale=1/255`)
- Class balancing using class weights
- Custom CNN baseline with BatchNorm, Dropout, and L2 regularization
- Transfer learning with EfficientNetB0 (ImageNet pretraining)
- Two-stage transfer training:
  - Stage 1: freeze backbone, train classifier head
  - Stage 2: unfreeze top backbone layers, fine-tune with low LR
- Adam optimizer with LR tuning
- Early stopping + ReduceLROnPlateau
- 60-epoch training configuration (`config.py`)

## Model Architecture Diagram (Explanation)

### Custom CNN

`Input(48x48x1) -> [Conv + BN] x N -> MaxPool -> Dropout -> GAP -> Dense -> Softmax(7)`

### EfficientNet Transfer

`Input(224x224x3) -> EfficientNetB0 Backbone -> GlobalAveragePooling -> BatchNorm -> Dense(256) -> Dropout -> Softmax(7)`

This hybrid strategy gives a robust baseline (CNN) and a high-capacity transfer model (EfficientNet), which is typically better for FER validation accuracy.

## Project Structure (Core)

```text
facial_emotion_detection/
|-- models/
|   |-- custom_cnn.py
|   |-- efficientnet_transfer.py
|-- preprocessing/
|   |-- data_preprocessing.py
|-- train_model.py
|-- predict.py
|-- realtime_emotion_detection.py
|-- config.py
|-- requirements.txt
|-- best_model.h5
|-- training_history.png
|-- artifacts/
    |-- cnn_model.h5
    |-- efficientnet_model.h5
    |-- confusion_matrix.png
    |-- classification_report.txt
    |-- training_log.txt
```

## Dataset Preparation

Expected layout:

```text
dataset_root/
|-- train/
|   |-- angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
|-- test/
    |-- angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

Dataset path priority:

1. `FER_DATASET_DIR` environment variable
2. `./backend/dataset`
3. `./dataset`

## Setup

```bash
pip install -r backend/requirements.txt
```

## Train

```bash
python backend/scripts/train_model.py
```

Generated outputs:

- `best_model.h5`
- `training_history.png`
- `artifacts/confusion_matrix.png`
- `artifacts/classification_report.txt`
- `artifacts/training_log.txt`

## Evaluate Single Image

```bash
python backend/scripts/predict.py --image path/to/face.jpg --model backend/artifacts/best_model.h5 --model-type auto
```

`--model-type` supports: `auto`, `cnn`, `resnet`, `transfer`, `efficientnet`

## Real-Time Webcam Detection

```bash
python backend/scripts/realtime_emotion_detection.py --model backend/artifacts/best_model.h5 --model-type auto
```

Press `q` to quit.

## Final-Year Submission Notes

- Include training curves and confusion matrix in your report.
- Mention two-stage fine-tuning and class balancing as core optimization steps.
- Report both validation and test accuracy from `artifacts/training_log.txt`.
