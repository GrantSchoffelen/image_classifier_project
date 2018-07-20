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


def main():
    args = get_input_args()

    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    image_sets, data_loaders = load_data(data_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model = models[args.arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if args.arch === 'densenet121':
        input_size = 1024

    if args.arch === 'vgg16':
        input_size = 784

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, args.hidden_units)),
                              ('relu', nn.ReLU()),
                              ('drop', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    training_and_testing_saving(model, epochs, optimizer, criterion, args.gpu, image_sets, data_loaders, args.hidden_units, input_size)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, default = "flowers",
                    help="data_directory")

    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='architecture model densenet121 or vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='hidden units')
    parser.add_argument('--epochs', type=int, default=3,
                        help='epochs')
    parser.add_argument('--gpu', action='store_true',
                        default=False, help='use gpu')

    return parser.parse_args()

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_sets = {}
    image_sets['train_data'] = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    image_sets['valid_data'] = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    image_sets['test_data'] = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    data_loaders = {}
    data_loaders['trainloader'] = torch.utils.data.DataLoader(image_sets['train_data'], batch_size=64, shuffle=True)
    data_loaders['validloader'] = torch.utils.data.DataLoader(image_sets['valid_data'], batch_size=32)
    data_loaders['testloader'] = torch.utils.data.DataLoader(image_sets['test_data'], batch_size=32)

    return image_sets, data_loaders

def training_and_testing_saving(model, epochs, optimizer, criterion, gpu,
                                image_sets, data_loaders, hidden_units, save_dir):
    print_every = 20
    steps = 0

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        devide = torch.device('cpu')

    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (images, labels) in enumerate(data_loaders['trainloader']):
            steps += 1

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in data_loaders['validloader']:
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/len(image_sets['train_data'])),
                      "Validation Loss: {:.3f}".format(valid_loss/len(data_loaders['validloader'])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(data_loaders['validloader'])))


    print('trained')
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loaders['testloader']:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    model.class_to_idx = data_loaders['train_data'].class_to_idx

    checkpoint = {
        'input_size': input_size,
        'output_size': 102,
        'class_to_idx': model.class_to_idx,
        'hidden_layer': hidden_units,
        'state_dict': model.state_dict(),
        'arch': 'densenet121'
    }

    torch.save(checkpoint, save_dir+'checkpoint.pth')

if __name__ == "__main__":
    main()
