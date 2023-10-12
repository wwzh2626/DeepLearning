import os

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from module import  AlexNet
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 导入image
img_path = './dataset/predict/1.jpeg'
assert os.path.exists(img_path), f"file:{img_path} is not exist."
img = Image.open(img_path)
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

json_path = "./class_indices.json"
assert os.path.exists(json_path), f"file:{json_path} is not exist."

with open(json_path, "r") as f:
    class_indict = json.load(f)

# 加载模型
model = AlexNet(num_classes=5).to(device)

# 加载权重
weight_path = "./AlexNet.pth"
assert os.path.exists(weight_path), f"file:{weight_path} is not exist."
model.load_state_dict(torch.load(weight_path))

# 开始预测
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0) # 得到每个种类的概率
    predict_cla = torch.argmax(predict).numpy()


print_res = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla].numpy()}"

plt.title(print_res)
for i in range(len(predict)):
    print(f"class: {class_indict[str(i)]}   prob: {predict[i].numpy()}")
plt.show()