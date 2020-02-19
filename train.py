import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from models import *

# from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
writer = SummaryWriter('runs/demo')


def save_checkpoint(path = './checkpoint/cifar_senet18.pth'):
    torch.save(model.state_dict(), path)
    print('saving model successful!')


# data
def data_preprocess():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 1)

    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('data has prepared.')
    return trainloader, testloader, classes


def train(epoch, model, criterion, optimizer, trainloader):
    # print('\nEpoch: ', epoch + 1)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if batch_idx % 10 == 0:
            print('train child epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx,
                                                                                      len(trainloader),
                                                                                      train_loss / (batch_idx + 1),
                                                                                      100. * correct / total))
            writer.add_scalar('training_loss', train_loss / (batch_idx + 1), epoch)
            writer.add_scalar('accuracy', 100. * correct / total, epoch)


if __name__ == '__main__':

    trainloader, testloader, classes = data_preprocess()

    model = resnet101(add_module = 'gc')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)

    for epoch in range(20):
        train(epoch, model, criterion, optimizer, trainloader)
    print('training finished.')
    save_checkpoint()
