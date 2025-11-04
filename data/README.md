# Data Module

This folder contains all datasets and data processing utilities for the Music Emotion Genre Recommender project.

## Directory Structure

```
data/
├── raw/                    # Raw audio files and datasets
│   ├── audio/             # Music audio files
│   └── emotion/           # Emotion detection datasets
├── processed/             # Preprocessed data
│   ├── spectrograms/      # Audio spectrograms
│   ├── mfcc/              # MFCC features
│   └── emotion_vectors/   # Processed emotion data
├── models/                # Saved trained models
└── mappings/              # Emotion-genre mapping files
```

## Datasets Required

### Audio Datasets
- **GTZAN Dataset**: 1000 audio tracks (10 genres, 100 tracks each)
- **FMA (Free Music Archive)**: Large-scale music dataset

### Emotion Datasets
- **FER-2013**: Facial emotion recognition dataset
- **CK+ (Extended Cohn-Kanade)**: Facial expression dataset

## Setup Instructions

1. Download datasets from official sources
2. Place raw audio files in `raw/audio/`
3. Place emotion datasets in `raw/emotion/`
4. Run preprocessing scripts from the root directory

## Note

Due to size constraints, datasets are not included in this repository. Please download them separately from their official sources.
