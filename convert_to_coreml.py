
import torch
import torch.nn as nn
import coremltools as ct
import os

# ============================================================================
# MNIST CNN MODEL (Must match live.py)
# ============================================================================

class MNIST_CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
        Conv2D(1->32) -> ReLU -> MaxPool2D(2x2)
        Conv2D(32->64) -> ReLU -> MaxPool2D(2x2)
        Flatten -> Dense(3136->128) -> ReLU -> Dropout(0.2)
        Dense(128->10) -> Softmax
    """
    
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)

def convert():
    model_path = "mnist_cnn.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    # Load PyTorch model
    model = MNIST_CNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Loaded model state dict.")
    except Exception as e:
        print(f"Failed to load state dict: {e}")
        return

    model.eval()

    # Trace the model
    dummy_input = torch.rand(1, 1, 28, 28)
    traced_model = torch.jit.trace(model, dummy_input)
    print("Traced model.")

    # Convert to CoreML
    # Inputs: name "image" (or similar), shape (1, 1, 28, 28)
    # Outputs: "probabilities" (1, 10)
    
    # We want the input to be an image if possible, but the original model takes (1, 1, 28, 28).
    # CoreML can accept ImageType if we specify it. 
    # However, since the Python code manually preprocesses (normalize), 
    # we might want to keep it as MultiArray input to control preprocessing in Swift, 
    # OR we can bake normalization into the CoreML model.
    # The Python code does:
    # img = digit_image.astype(np.float32) / 255.0
    # img = (img - Config.MNIST_MEAN) / Config.MNIST_STD
    # MNIST_MEAN = 0.1307, MNIST_STD = 0.3081
    
    scale = 1/(0.3081*255.0)
    bias = -0.1307/0.3081
    
    # Actually, simpler to just let Swift handle normalization or use the ImageType with bias/scale.
    # let's try to use ImageType for convenience in Swift.
    # Input is grayscale (1 channel).
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_1", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="output_1")],
        minimum_deployment_target=ct.target.macOS14
    )
    
    save_path = "MNIST_CNN.mlpackage"
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
        
    mlmodel.save(save_path)
    print(f"Saved CoreML model to {save_path}")

if __name__ == "__main__":
    convert()
