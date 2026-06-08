# Sign Language Recognition Model

A real-time sign language recognition system built using OpenCV, MediaPipe, and Machine Learning.

The application detects hand landmarks from a webcam feed, extracts spatial features, and classifies sign language gestures in real time. To improve visual clarity and focus on gesture recognition, the system displays a black background with hand landmark visualization instead of the raw camera feed.

## Features

* Real-time hand tracking using MediaPipe
* Sign language gesture recognition
* Confidence score display
* Live text generation from recognized signs
* Text-to-speech conversion
* Minimalist black-background visualization
* Machine learning based classification

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy
* Pyttsx3

## Project Workflow

Webcam Input

↓

Hand Landmark Detection (MediaPipe)

↓

Feature Extraction (x, y coordinates)

↓

Model Prediction

↓

Text Generation

↓

Speech Output

## Controls

| Key | Action               |
| --- | -------------------- |
| S   | Speak generated text |
| C   | Clear generated text |
| Q   | Quit application     |

## Current Supported Signs

* A
* B
* C
* D
* E

## Installation

```bash
git clone https://github.com/rabbitx07/computervision.git
cd computervision

pip install -r requirements.txt
```

## Run

Collect Dataset:

```bash
python collect_data.py
```

Train Model:

```bash
python train_model.py
```

Start Prediction:

```bash
python predict.py
```

## Future Improvements

* Full A-Z sign recognition
* Word prediction and auto-complete
* Improved gesture stabilization
* Deep learning based classification
* Web application interface



## Author

Anshika
BCA (AI & Data Science)
Graphic Era University
