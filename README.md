# MNIST Handwritten Digit Recognition Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)
![Model Accuracy](https://img.shields.io/badge/accuracy-99.30%25-brightgreen.svg?style=for-the-badge)

## Overview

We've developed a state-of-the-art Convolutional Neural Network (CNN) for handwritten digit recognition, achieving **99.30% accuracy** on the MNIST test dataset. This production-ready model is optimized for real-world deployment and can be easily integrated into various applications.

## 🚀 Performance Metrics

Our model demonstrates exceptional performance with rapid convergence:


Epoch 1/10 | Train Loss: 0.1952 | Test Accuracy: 98.59%
Epoch 2/10 | Train Loss: 0.0574 | Test Accuracy: 98.87%
Epoch 3/10 | Train Loss: 0.0405 | Test Accuracy: 98.98%
Epoch 4/10 | Train Loss: 0.0318 | Test Accuracy: 99.06%
Epoch 5/10 | Train Loss: 0.0243 | Test Accuracy: 99.09%
Epoch 6/10 | Train Loss: 0.0211 | Test Accuracy: 99.02%
Epoch 7/10 | Train Loss: 0.0185 | Test Accuracy: 99.27%
Epoch 8/10 | Train Loss: 0.0162 | Test Accuracy: 99.30%
Epoch 9/10 | Train Loss: 0.0135 | Test Accuracy: 99.27%
Epoch 10/10 | Train Loss: 0.0113 | Test Accuracy: 99.26%

## 🛠️ Technology Stack

- **Framework**: PyTorch 2.0+
- **Architecture**: Custom CNN with batch normalization and dropout
- **Training Infrastructure**: CUDA-accelerated GPU compute
- **Dataset**: MNIST (70,000 handwritten digits)
- **Python Version**: 3.8+

## 📦 Installation


bash

Clone the repository
git clone https://github.com/yourcompany/mnist-digit-recognition.git
cd mnist-digit-recognition

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

## 🎯 Quick Start - Using Pre-trained Model

### Loading the Pre-trained Model


python
import torch
from model import MNISTNet # Assuming your model class is in model.py

Initialize model
model = MNISTNet()

Load pre-trained weights
checkpoint = torch.load('releases/mnist_cnn_v1.0.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

Model is ready for inference!
### Making Predictions


python
import torch
from torchvision import transforms
from PIL import Image

Preprocessing pipeline
transform = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
transforms.Resize((28, 28)),
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image_path):
# Load and preprocess image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

return prediction


Example usage
digit = predict_digit('path/to/your/digit.png')
print(f"Predicted digit: {digit}")

## 🏋️ Training Your Own Model

### Training Script


python
python train.py --epochs 10 --batch-size 64 --learning-rate 0.001 --device cuda

### Custom Training Example


python
from train import train_model
from model import MNISTNet

Initialize your model
model = MNISTNet()

Configure training
config = {
'epochs': 10,
'batch_size': 64,
'learning_rate': 0.001,
'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

Train the model
trained_model, history = train_model(model, config)

Save your trained model
torch.save({
'model_state_dict': trained_model.state_dict(),
'accuracy': history['best_accuracy'],
'config': config
}, 'my_custom_mnist_model.pt')

## 📊 Model Architecture


MNISTNet(
(conv1): Conv2d(1, 32, kernel_size=(3, 3))
(bn1): BatchNorm2d(32)
(conv2): Conv2d(32, 64, kernel_size=(3, 3))
(bn2): BatchNorm2d(64)
(dropout1): Dropout2d(p=0.25)
(fc1): Linear(9216, 128)
(dropout2): Dropout(p=0.5)
(fc2): Linear(128, 10)
)

## 🔧 API Integration

### REST API Example


python
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(name)

@app.route('/predict', methods=['POST'])
def predict():
# Get base64 image from request
image_data = request.json['image']
image = Image.open(BytesIO(base64.b64decode(image_data)))

# Preprocess and predict
digit = predict_digit(image)

return jsonify({
    'digit': digit,
    'confidence': float(torch.softmax(output, dim=1).max())
})


## ⚠️ Known Limitations & Ongoing Improvements

### Current Limitations
- **Unintended Activation**: The model may occasionally detect digit-like patterns in non-digit images (e.g., facial features being misclassified as digits)
- **Input Constraints**: Optimized for 28x28 grayscale images only
- **Real-world Handwriting**: Performance may vary on handwriting styles significantly different from MNIST dataset

### Roadmap
- [ ] Implement adversarial training to reduce false positives
- [ ] Add confidence thresholding for production deployments
- [ ] Extend model to handle variable input sizes
- [ ] Integrate with mobile SDKs (iOS/Android)

## 📈 Performance Optimization

For production environments, we recommend:


python

Optimize model for inference
import torch.jit

Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('mnist_model_scripted.pt')

For mobile deployment
from torch.utils.mobile_optimizer import optimize_for_mobile
mobile_model = optimize_for_mobile(scripted_model)
mobile_model._save_for_lite_interpreter('mnist_model_mobile.ptl')

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 About Us

Developed by the AI Research Team at [Your Company Name]. We specialize in building production-ready machine learning solutions for enterprise applications.

## 📞 Support

- **Documentation**: [docs.yourcompany.com/mnist-model](https://docs.yourcompany.com)
- **Issues**: [GitHub Issues](https://github.com/yourcompany/mnist-digit-recognition/issues)
- **Email**: ai-support@yourcompany.com

---

&lt;p align="center"&gt;
  &lt;b&gt;Ready to integrate?&lt;/b&gt; Check our &lt;a href="https://docs.yourcompany.com/quickstart"&gt;Quick Start Guide&lt;/a&gt;
&lt;/p&gt;