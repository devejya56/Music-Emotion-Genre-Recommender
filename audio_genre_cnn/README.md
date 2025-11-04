# Audio Genre CNN Module

This module handles music genre classification using Convolutional Neural Networks (CNN) on audio spectrograms.

## Directory Structure

```
audio_genre_cnn/
├── models/
│   ├── cnn_architecture.py    # CNN model definition
│   ├── train.py               # Training script
│   └── evaluate.py            # Model evaluation
├── preprocessing/
│   ├── audio_processor.py     # Audio preprocessing
│   └── feature_extractor.py   # MFCC, spectrogram extraction
├── utils/
│   └── data_loader.py         # Data loading utilities
└── config.py                  # Configuration parameters
```

## Key Features

- **Mel-Spectrogram Generation**: Convert audio signals into visual representations
- **MFCC Feature Extraction**: Extract Mel-Frequency Cepstral Coefficients
- **CNN Architecture**: Deep learning model for genre classification
- **Target Accuracy**: 85%+

## Supported Genres

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## Model Architecture

```python
# Simplified CNN architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

## Usage

```bash
# Train the model
python models/train.py --epochs 50 --batch_size 32

# Evaluate the model
python models/evaluate.py --model_path ./saved_models/genre_classifier.h5

# Preprocess audio files
python preprocessing/audio_processor.py --input_dir ../data/raw/audio
```

## Dependencies

- TensorFlow/Keras
- Librosa
- NumPy
- Matplotlib
- SciPy

## Performance Metrics

- Training Accuracy: Target 90%+
- Validation Accuracy: Target 85%+
- Test Accuracy: Target 85%+
