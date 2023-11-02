import torch
from model import AlexNet
from torchvision import transforms, datasets, utils
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

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

data_root = os.path.dirname(os.getcwd())
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
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4,
                                              shuffle=True)
#
# test_image, test_label = next(iter(validata_loader))
#
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(" ".join("%5s" % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
pata = list[net.parameters()]
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = './AlexNet.pth'
best_accuracy = 0.0
train_steps = len(train_loader)
epochs = 10


for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)

    for step, data in enumerate(train_bar, start=0):
        images, label = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, label.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_accuracy:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')
