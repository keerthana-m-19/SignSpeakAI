# 🤟 SignSpeakAI — Real-Time Sign Language to Speech Translator

> An AI-powered application that translates **American Sign Language (ASL)** hand gestures into text and speech in real time using deep learning and computer vision.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖐️ **Real-Time Hand Detection** | MediaPipe Hands with landmark tracking and dynamic ROI extraction |
| 🧠 **Deep Learning Classification** | MobileNetV2 transfer learning for 29 ASL classes (A-Z + space/del/nothing) |
| 📝 **Word Formation** | Stabilized predictions with buffer — consecutive identical predictions form letters into words |
| 🔊 **Text-to-Speech** | Non-blocking async speech output via pyttsx3 |
| 🎤 **Speech-to-Text** | Optional bidirectional communication via Google Speech Recognition |
| 📊 **Professional UI** | Real-time HUD overlay with confidence bars, FPS counter, and keyboard controls |
| 📸 **Data Collection** | Built-in webcam data collection tool with progress tracking |
| 📈 **Training Pipeline** | Two-phase transfer learning with data augmentation and training history plots |

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        SignSpeakAI                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Webcam Feed                                                │
│       │                                                     │
│       ▼                                                     │
│  MediaPipe Hand Detection                                   │
│       │                                                     │
│       ▼                                                     │
│  Hand ROI Extraction (padded, squared)                      │
│       │                                                     │
│       ▼                                                     │
│  MobileNetV2 Feature Extractor (ImageNet pretrained)        │
│       │                                                     │
│       ▼                                                     │
│  Dense Classification Head (29 classes)                     │
│       │                                                     │
│       ▼                                                     │
│  Prediction Stabilizer (N/buffer consensus)                 │
│       │                                                     │
│       ▼                                                     │
│  Word Formation Engine (space / del / nothing handling)     │
│       │                                                     │
│       ▼                                                     │
│  Text-to-Speech (pyttsx3, async)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
SignSpeakAI/
│
├── dataset/
│   ├── train/               # Training images (29 class folders)
│   │   ├── A/ ... Z/
│   │   ├── del/
│   │   ├── nothing/
│   │   └── space/
│   └── test/                # Test images
│
├── models/
│   ├── sign_model.keras     # Trained MobileNetV2 model
│   ├── class_names.npy      # Saved class names array
│   └── training_history.png # Training accuracy/loss curves
│
├── src/
│   ├── __init__.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_model.py   # MobileNetV2 transfer learning pipeline
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── realtime_prediction.py  # SignPredictor + PredictionStabilizer
│   │
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── text_to_speech.py       # Async TTS engine
│   │   └── speech_to_text.py       # Google STT engine
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── hand_detector.py        # MediaPipe hand detector
│   │
│   └── data_collection/
│       ├── __init__.py
│       └── collect_data.py         # Webcam data collection tool
│
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **Webcam** (built-in or external)
- **GPU** (recommended for training, not required for inference)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/keerthana-m-19/SignSpeakAI
cd SignSpeakAI

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and extract it:

```text
dataset/
├── train/       # ~87,000 images across 29 classes
│   ├── A/
│   ├── B/
│   └── ...
└── test/
```

---

## 🎓 Training the Model

```bash
python -m src.training.train_model
```

This runs a **two-phase transfer learning** pipeline:

| Phase | Epochs | Learning Rate | Description |
|---|---|---|---|
| Phase 1: Feature Extraction | 20 | 0.001 | MobileNetV2 base frozen, only classification head trains |
| Phase 2: Fine-Tuning | 10 | 0.0001 | Top 30 base layers unfrozen for end-to-end refinement |

**Data augmentation** applied during training:
- Random horizontal flip
- Random rotation (±15°)
- Random zoom (±15%)
- Random translation (±10%)
- Random brightness (±20%)
- Random contrast (±20%)

**Callbacks**:
- `EarlyStopping` — stops training if validation accuracy plateaus
- `ReduceLROnPlateau` — halves learning rate on validation loss plateau
- `ModelCheckpoint` — saves only the best model

**Output**:
- `models/sign_model.keras` — trained model
- `models/class_names.npy` — class labels
- `models/training_history.png` — accuracy/loss plots

---

## 🖥️ Running the Application

```bash
python main.py
```

### Controls

| Key | Action |
|---|---|
| `S` | Speak accumulated text aloud |
| `C` | Clear text buffer |
| `Q` | Quit application |

### UI Display

The application shows a professional HUD overlay with:
- **Title bar** with project name
- **Prediction panel** — detected letter, confidence %, top-3 predictions
- **Visual confidence bar** — color-coded (green/yellow/red)
- **Text panel** — accumulated word/sentence
- **Controls bar** — keyboard shortcuts
- **FPS counter** — real-time performance metric

---

## 📸 Collecting Custom Data

```bash
python -m src.data_collection.collect_data --label A --count 200
```

| Argument | Description | Default |
|---|---|---|
| `--label`, `-l` | Class label to collect | *required* |
| `--count`, `-c` | Number of images | 200 |
| `--output`, `-o` | Output directory | `dataset/train` |
| `--size`, `-s` | Image size | 224 |

**Controls during collection:**
- `SPACE` — Start/pause capture
- `Q` — Quit

---

## 🎯 Performance Targets

| Metric | Target | Description |
|---|---|---|
| Training Accuracy | > 95% | On training set |
| Validation Accuracy | > 92% | On held-out validation set |
| Real-time Accuracy | > 85% | During live webcam usage |

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| **TensorFlow / Keras** | Deep learning framework |
| **MobileNetV2** | Pre-trained feature extractor (ImageNet) |
| **MediaPipe** | Real-time hand detection & landmark tracking |
| **OpenCV** | Webcam capture & UI rendering |
| **pyttsx3** | Offline text-to-speech |
| **SpeechRecognition** | Speech-to-text (Google API) |
| **NumPy** | Numerical computing |
| **Matplotlib** | Training visualization |

---

## 🔮 Future Enhancements

- [ ] Full sentence generation with NLP
- [ ] Multi-language speech output
- [ ] Indian Sign Language (ISL) support
- [ ] Web application deployment (Flask/Streamlit)
- [ ] Mobile app deployment (TFLite)
- [ ] Save and export translated conversations
- [ ] Two-way communication mode
- [ ] Gesture-based word shortcuts

---

## 📄 License

This project is developed as a final-year AI portfolio project. Feel free to use and modify for educational purposes.

---

<p align="center">
  Built with ❤️ using TensorFlow, MediaPipe, and OpenCV
</p>
