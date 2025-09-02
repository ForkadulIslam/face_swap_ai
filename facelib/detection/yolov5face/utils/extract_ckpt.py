import torch
import sys
import os

# Setup dynamic device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sys.path.insert(0, './facelib/detection/yolov5face')

# Load the model to the selected device
ckpt = torch.load('facelib/detection/yolov5face/yolov5n-face.pt', map_location=device)
model = ckpt['model'].to(device)

# Save only the weights
os.makedirs('weights/facelib', exist_ok=True)
torch.save(model.state_dict(), 'weights/facelib/yolov5n-face.pth')