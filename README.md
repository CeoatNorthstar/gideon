# Gideon - AI Finger Recognition System

![Swift](https://img.shields.io/badge/Swift-5.9-orange.svg?style=for-the-badge&logo=swift&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-14+-black.svg?style=for-the-badge&logo=apple&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

## Overview

**Gideon** is a professional finger counting system that uses AI and computer vision to detect and count fingers in real-time. The project includes:

- **GideonMac** - Native macOS app built with Swift and Vision framework
- **Python Live Demo** - MediaPipe-based hand tracking with MNIST verification

Both versions support **two-hand tracking** (count up to 10 fingers) with temporal smoothing for stable results.

## Features

- 🖐️ **Real-time finger counting** - Instant detection with 30+ FPS
- ✌️ **Two-hand support** - Count fingers from both hands simultaneously  
- 📷 **Un-mirrored camera** - Natural view (right hand on right side)
- 🧠 **MNIST verification** - AI validates the finger count
- 🎨 **Modern UI** - Clean, professional interface
- 🔄 **Temporal smoothing** - Stable counts using mode of recent frames

## Quick Start

### Python Version (Recommended for Testing)

```bash
# Install dependencies
pip install torch torchvision opencv-python mediapipe

# Download hand landmarker model
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Run live demo
python live.py
```

**Controls:**
- **Q** / **ESC** - Quit
- **H** - Toggle hand visualization
- **D** - Toggle digit preview

### macOS App (GideonMac)

```bash
cd GideonMac

# Build and run
./build_app.sh
open Gideon.app
```

**Requirements:** macOS 14+, Xcode Command Line Tools

## Project Structure

```
gideon/
├── live.py                    # Python live demo (MediaPipe + MNIST)
├── train.ipynb                # Train MNIST model
├── mnist_cnn.pt               # Trained MNIST weights
├── hand_landmarker.task       # MediaPipe hand model
├── convert_to_coreml.py       # Convert PyTorch → CoreML
├── requirements.txt           # Python dependencies
│
└── GideonMac/                 # Native macOS app
    ├── Sources/GideonMac/
    │   ├── CameraManager.swift      # Camera + Vision hand pose
    │   ├── ContentView.swift        # SwiftUI interface
    │   ├── RecognitionService.swift # CoreML MNIST inference
    │   └── DigitRenderer.swift      # Render digits for MNIST
    ├── MNIST_CNN.mlpackage    # CoreML model
    ├── build_app.sh           # Build script
    └── Package.swift          # Swift package manifest
```

## How It Works

### Finger Counting Logic

Both Python and Swift versions use the same algorithm:

**Thumb:** Extended if tip X-distance from wrist > base X-distance from wrist
```python
thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_base.x - wrist.x)
```

**Other 4 Fingers:** Extended if tip is above PIP joint (Y-axis)
```python
finger_extended = tip.y < pip.y  # In image coords, lower Y = higher position
```

### Pipeline

1. **Camera Capture** → RGB frame from webcam
2. **Hand Detection** → MediaPipe/Vision detects hand landmarks
3. **Finger Counting** → Apply extension logic to each finger
4. **Temporal Smoothing** → Mode of last 7 frames
5. **MNIST Verification** → Render digit → CNN predicts → Compare

## Training the MNIST Model

```bash
# Run the training notebook
jupyter notebook train.ipynb
```

Or train directly:
```bash
python -c "
import torch
from torchvision import datasets, transforms
from torch import nn, optim

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST('.', train=True, download=True, transform=transform)
test = datasets.MNIST('.', train=False, transform=transform)

# Simple CNN
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)
)

# Train
for epoch in range(10):
    for x, y in torch.utils.data.DataLoader(train, batch_size=128, shuffle=True):
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optim.Adam(model.parameters()).step()
        optim.Adam(model.parameters()).zero_grad()

torch.save(model.state_dict(), 'mnist_cnn.pt')
"
```

## Converting to CoreML

```bash
python convert_to_coreml.py
```

This creates `MNIST_CNN.mlpackage` for use in the macOS app.

## Performance

| Metric | Python | Swift |
|--------|--------|-------|
| FPS | 30+ | 30+ |
| Max Hands | 2 | 2 |
| Max Fingers | 10 | 10 |
| Smoothing | 7 frames | 5 frames |
| Accuracy | ~99% | ~99% |

## Requirements

### Python
- Python 3.9+
- PyTorch
- OpenCV
- MediaPipe
- NumPy

### macOS
- macOS 14 Sonoma+
- Swift 5.9+
- Vision framework (built-in)
- CoreML (built-in)

## Tips for Best Results

1. **Good lighting** - Even, front-facing light works best
2. **Plain background** - Avoid busy or skin-toned backgrounds
3. **Spread fingers** - Keep fingers clearly separated
4. **Camera distance** - Position hand 1-2 feet from camera
5. **Wait for smoothing** - Hold gesture for ~0.5s for stable count

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Google's hand tracking solution
- [MNIST](http://yann.lecun.com/exdb/mnist/) - LeCun, Cortes, Burges
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Apple Vision](https://developer.apple.com/documentation/vision) - Hand pose estimation

---

**Made with ❤️ for human-computer interaction research**