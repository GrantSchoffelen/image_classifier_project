
import argparse
from os import listdir
import torch
import json

from PIL import Image
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import torch.nn.functional as F
from torch.autograd import Variable


def main():
    in_arg = get_input_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        devide = torch.device('cpu')

    image_path = args.input
    trained_model = load_checkpoint(args.checkpoint)
    cat_to_name = get_label_mapping(args.category_names)

    probs, classes = predict('flowers/test/10/image_07090.jpg', trained_model, args.top_k)

    classes = classes.astype(str)

    class_names = []
    for i in classes:
        class_names.append(cat_to_name[i])
    print(class_names)
    print(probs)

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, default=None,
                    help='input image file')
    parser.add_argument('checkpoint', type=str, default=None,
                    help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=3,
                        help='return top k most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='json map of names')
    parser.add_argument('--gpu', action='store_true',
                        help='use gpu')

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
    new_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = new_classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model

def get_label_mapping(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def process_image(image):
    img = Image.open(image)

    img.thumbnail((256, 256))
    img=img.crop((16, 16, 240, 240))

    np_image = np.array(img)/255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    img = (np_image - means)/stds

    tranposed_img = img.transpose(2, 0, 1)

    tensor = torch.from_numpy(tranposed_img)

    return tensor

def predict(image_path, model, topk=5):
    img = process_image(image_path)
    model.eval()
    img.unsqueeze_(0)
    output = model.forward(img.float())
    ps = F.softmax(output, dim=1)

    probs, classes = ps.topk(5)
    probs = probs.detach().numpy()[0]
    classes = classes.numpy()[0]
    return probs, classes



if __name__ == "__main__":
    main()
