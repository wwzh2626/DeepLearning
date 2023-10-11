import os
from PIL import Image
import torch
from torchvision import transforms
from module import  AlexNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

img_path = './dataset/predict/1.jpeg'
assert os.path.exists(img_path), f"file:{img_path} is not exist."
img = Image.open(img_path)