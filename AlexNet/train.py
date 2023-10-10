import torch
from module import AlexNet
from torchvision import transforms, datasets, utils
import numpy as np
import torch.optim as optim
import os
import json
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

data_root = os.getcwd()
image_path = data_root + "/dataset/flower_data"
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
print(train_num)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', "w") as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True)

validate_dataset = datasets.ImageFolder(root=image_path + '/val',
                                        transform=data_transform['val'])

val_num = len(validate_dataset)
validata_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4,
                                        shuffle=True)

test_data_iter = iter(validata_loader)
test_image, test_label = test_data_iter.next()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(" ".join("%5s" % cla_dict[test_label[j].item()] for j in range(4)))
imshow(utils.make_grid(test_image))

