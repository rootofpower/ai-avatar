import time
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights).to(device)
model.eval()

preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = np.zeros((512, 512, 3), dtype=np.uint8)
image = cv2.circle(image, (256, 256), 100, (255, 255, 255), -1)

t0 = time.time()
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)
t1 = time.time()
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)
t2 = time.time()

print("Prep:", t1 - t0)
print("Inference:", t2 - t1)
print("Shape:", output_predictions.shape)
