# Music Emotion Genre Recommender

> Music genre classification and recommendation system based on user emotions using deep learning

## 📋 Problem Statement

Music plays a crucial role in influencing and reflecting human emotions. However, finding music that matches one's current emotional state can be challenging. This project addresses two key problems:

1. **Emotion-Based Music Discovery**: Users often struggle to find music that resonates with their current emotional state
2. **Intelligent Genre Classification**: Automated classification of music into genres based on audio features
3. **Personalized Recommendations**: Bridging the gap between detected emotions and suitable music genres

The system aims to:
- Detect user emotions through facial expressions or text input
- Classify music tracks into appropriate genres using audio analysis
- Provide personalized music recommendations that align with the user's emotional state

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Streamlit Web App)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Emotion Detection Module                    │
│              (Facial Expression / Text Input)                │
│         - CNN for facial emotion recognition                 │
│         - NLP for text-based emotion analysis                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Detected Emotion     │
          │  (Happy, Sad, etc.)   │
          └───────────┬───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Recommender Engine Module                       │
│     - Emotion-to-genre mapping algorithm                     │
│     - Collaborative filtering                                │
│     - Content-based filtering                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Audio Genre Classification Module                 │
│              (CNN on Audio Spectrograms)                     │
│         - Feature extraction (MFCC, Mel-spectrogram)         │
│         - Deep learning classification                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Music Recommendations│
          │  with Genre Labels    │
          └───────────────────────┘
```

## 📊 Datasets

### 1. **Audio Dataset**
- **GTZAN Dataset**: 1000 audio tracks (10 genres, 100 tracks each)
  - Genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **FMA (Free Music Archive)**: Large-scale music dataset
- **Million Song Dataset** (optional): For enhanced training

### 2. **Emotion Dataset**
- **FER-2013**: 35,887 grayscale facial images (48x48 pixels)
  - Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **CK+ (Extended Cohn-Kanade)**: Facial expression dataset
- **EmoMusic**: Emotion-labeled music dataset

### 3. **Emotion-Genre Mapping**
- Custom dataset mapping emotions to music genres
- User feedback data for recommendation refinement

## 🛠️ Technology Stack

### **Deep Learning & ML**
- **TensorFlow / Keras**: Deep learning model development
- **PyTorch**: Alternative DL framework
- **scikit-learn**: Traditional ML algorithms and preprocessing
- **OpenCV**: Image processing for facial emotion detection

### **Audio Processing**
- **Librosa**: Audio analysis and feature extraction
- **PyDub**: Audio file manipulation
- **SciPy**: Scientific computing for signal processing

### **Web Framework**
- **Streamlit**: Interactive web application
- **Flask/FastAPI**: Backend API (optional)

### **Data Processing**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

### **Model Deployment**
- **Docker**: Containerization
- **Heroku/AWS**: Cloud deployment options

## 📦 Module Breakdown

### 1. **Data Module** (`/data`)
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

**Responsibilities:**
- Data collection and storage
- Data preprocessing pipelines
- Feature extraction utilities

### 2. **Audio Genre CNN Module** (`/audio_genre_cnn`)
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

**Key Features:**
- Mel-spectrogram generation
- MFCC feature extraction
- CNN architecture for genre classification
- Model training and evaluation
- Accuracy: Target 85%+

### 3. **Emotion Detection Module** (`/emotion_detection`)
```
emotion_detection/
├── models/
│   ├── facial_emotion_cnn.py  # CNN for facial expressions
│   ├── text_emotion_nlp.py    # NLP for text-based emotion
│   └── train.py               # Training script
├── preprocessing/
│   ├── face_detector.py       # Face detection and preprocessing
│   └── text_processor.py      # Text preprocessing
├── utils/
│   └── emotion_mapper.py      # Emotion categorization
└── config.py                  # Configuration parameters
```

**Key Features:**
- Real-time facial emotion detection using webcam
- Text-based emotion analysis (optional)
- 7 emotion categories: Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral
- Pre-trained model fine-tuning

### 4. **Recommender Engine Module** (`/recommender_engine`)
```
recommender_engine/
├── algorithms/
│   ├── emotion_genre_mapper.py   # Emotion to genre mapping
│   ├── collaborative_filter.py   # User-based recommendations
│   └── content_based_filter.py   # Content similarity
├── database/
│   ├── music_database.py         # Music catalog management
│   └── user_preferences.py       # User history and preferences
├── utils/
│   └── recommendation_utils.py   # Helper functions
└── config.py                     # Configuration parameters
```

**Key Features:**
- Emotion-to-genre mapping algorithm
- Hybrid recommendation system (collaborative + content-based)
- User preference learning
- Real-time recommendation generation

**Emotion-Genre Mapping Example:**
```python
emotion_genre_map = {
    'Happy': ['Pop', 'Dance', 'Disco', 'Reggae'],
    'Sad': ['Blues', 'Classical', 'Jazz'],
    'Angry': ['Metal', 'Rock', 'Hip-hop'],
    'Calm': ['Classical', 'Jazz', 'Ambient'],
    'Energetic': ['Electronic', 'Rock', 'Hip-hop']
}
```

### 5. **Streamlit UI Module** (`/streamlit_ui`) [Optional]
```
streamlit_ui/
├── app.py                    # Main Streamlit application
├── pages/
│   ├── emotion_capture.py    # Emotion detection interface
│   ├── music_player.py       # Music playback interface
│   └── recommendations.py    # Recommendation display
├── components/
│   ├── webcam.py            # Webcam integration
│   └── audio_player.py      # Audio player widget
└── assets/
    ├── styles.css           # Custom CSS
    └── images/              # UI images and icons
```

**Key Features:**
- Interactive web interface
- Real-time emotion detection via webcam
- Music playback functionality
- Recommendation visualization
- User feedback collection

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
OpenCV
Librosa
Streamlit
```

### Installation
```bash
# Clone the repository
git clone https://github.com/devejya56/Music-Emotion-Genre-Recommender.git
cd Music-Emotion-Genre-Recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Project Setup
```bash
# Download datasets (instructions in data/README.md)
python scripts/download_datasets.py

# Preprocess data
python scripts/preprocess_data.py

# Train genre classification model
cd audio_genre_cnn
python models/train.py

# Train emotion detection model
cd ../emotion_detection
python models/train.py

# Run the application
cd ../streamlit_ui
streamlit run app.py
```

## 📈 Model Performance Targets

| Model | Target Accuracy | Status |
|-------|----------------|--------|
| Genre Classification CNN | 85%+ | 🔄 In Progress |
| Facial Emotion Detection | 70%+ | 🔄 In Progress |
| Recommendation Precision | 80%+ | 🔄 In Progress |

## 🎯 Roadmap

- [x] Project structure setup
- [ ] Data collection and preprocessing
- [ ] Audio genre classification model
- [ ] Emotion detection model (facial)
- [ ] Recommender engine implementation
- [ ] Streamlit UI development
- [ ] Model integration and testing
- [ ] Performance optimization
- [ ] Deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or suggestions, please open an issue or contact the repository owner.

## 🙏 Acknowledgments

- GTZAN Dataset creators
- FER-2013 Dataset contributors
- TensorFlow and PyTorch communities
- Librosa audio processing library

---

**Note**: This is an active development project. Features and documentation will be updated regularly.
