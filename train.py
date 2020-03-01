import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Training')
    parser.add_argument('--model', type = str, help = 'network model')
    parser.add_argument('--module', type = str, default = None, help = 'add a module: se / gc / cbam')

    parser.add_argument('--lr', type = float, default = 0.1, help = 'learning rare')
    parser.add_argument('--num_epoch', type = int, default = 150, help = 'epoch')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch_size')

    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')

    args = parser.parse_args()
    return args


# data
def data_prepare(batch_size):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 1)

    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('data has prepared.')
    return trainloader, testloader, classes


def train(epoch, model, criterion, optimizer, trainloader, device):
    # print('\nEpoch: ', epoch + 1)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    # for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
    for inputs, labels in tqdm(trainloader):
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

    print('train child epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(epoch + 1, train_loss / len(trainloader),
                                                                       100. * correct / total))
    # writer.add_scalar('training_loss', train_loss / (batch_idx + 1), epoch)
    # writer.add_scalar('accuracy', 100. * correct / total, epoch)


def test(epoch, model, criterion, optimizer, testloader, device):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for step, (data, targets) in enumerate(testloader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            current_acc = 100. * correct / total

            print('test child epoch : {} [{}/{}]| acc: {:.3f}'.format(epoch, step, len(testloader), current_acc
                                                                      ))
    # save checkpoint
    if current_acc > best_acc:
        best_acc = max(current_acc, best_acc)
        checkpoint = {
            'best_acc': best_acc,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        model_path = os.path.join('checkpoint', 'best_checkpoint.pth.tar')
        torch.save(checkpoint, model_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    trainloader, testloader, classes = data_prepare(args.batch_size)

    module = args.module

    model = resnet101(add_module = module).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

    for epoch in range(args.num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epoch))
        train(epoch, model, criterion, optimizer, trainloader, device)
        test(epoch, model, criterion, optimizer, trainloader, device)
        scheduler.step()
    #     save_checkpoint('resnet_epoch_{}.pth'.format(epoch))
    # print('training finished.')


if __name__ == '__main__':
    main()
