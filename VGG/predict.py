import torch
import os
import json
from PIL import Image
from torchvision import transforms
from model import vgg
import matplotlib.pyplot as plt
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    data_root = os.path.dirname(os.getcwd())
    image_path = os.path.join(data_root, "dataset/predict/1.jpeg")
    assert os.path.exists(image_path), f"image_path: {image_path} is not exist ."
    img = Image.open(image_path)
    plt.imshow(img)
    # transforms image into [NxCxHxW]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    net = vgg("vgg16",num_classes=5)
    net.to(device)
    weight_path = "./VGGNet.pth"
    assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()


    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()
    print("fff")