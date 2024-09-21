'''Train CIFAR10 with PyTorch and ActiveConfusion Algorithm - Detailed Logging'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import csv

from models import *
from loader import Loader, RotationLoader
from utils import progress_bar
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with ActiveConfusion')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--k', default=5000, type=int, help='number of images to select for downstream task')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = RotationLoader(is_train=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = RotationLoader(is_train=False,  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(reduction='none')  # Changed to 'none' to get per-sample loss
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

def compute_confusion_score(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).float()
    return (1 - correct.mean().item()) * 4

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(trainloader):
        inputs, inputs1, inputs2, inputs3 = inputs.to(device), inputs1.to(device), inputs2.to(device), inputs3.to(device)
        targets, targets1, targets2, targets3 = targets.to(device), targets1.to(device), targets2.to(device), targets3.to(device)
        
        optimizer.zero_grad()
        outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs1, targets1)
        loss3 = criterion(outputs2, targets2)
        loss4 = criterion(outputs3, targets3)
        loss = (loss1 + loss2 + loss3 + loss4) / 4.
        loss.mean().backward()
        optimizer.step()

        train_loss += loss.mean().item()
        total += targets.size(0) * 4
        correct += (outputs.argmax(1) == targets).sum().item()
        correct += (outputs1.argmax(1) == targets1).sum().item()
        correct += (outputs2.argmax(1) == targets2).sum().item()
        correct += (outputs3.argmax(1) == targets3).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    image_data = []
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, paths) in enumerate(testloader):
            inputs, inputs1, inputs2, inputs3 = inputs.to(device), inputs1.to(device), inputs2.to(device), inputs3.to(device)
            targets, targets1, targets2, targets3 = targets.to(device), targets1.to(device), targets2.to(device), targets3.to(device)
            
            outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)
            
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            
            batch_loss = (loss1 + loss2 + loss3 + loss4) / 4.
            test_loss += batch_loss.mean().item()
            total += targets.size(0) * 4
            correct += (outputs.argmax(1) == targets).sum().item()
            correct += (outputs1.argmax(1) == targets1).sum().item()
            correct += (outputs2.argmax(1) == targets2).sum().item()
            correct += (outputs3.argmax(1) == targets3).sum().item()

            # Compute confusion score and store data for each image
            for i in range(targets.size(0)):
                confusion_score = compute_confusion_score(
                    torch.stack([outputs[i], outputs1[i], outputs2[i], outputs3[i]]),
                    torch.stack([targets[i], targets1[i], targets2[i], targets3[i]])
                )
                avg_loss = (loss1[i] + loss2[i] + loss3[i] + loss4[i]) / 4.
                image_data.append({
                    'path': paths[i],
                    'loss': avg_loss.item(),
                    'confusion_score': confusion_score
                })

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/rotation.pth')
        best_acc = acc

    return image_data

def active_confusion_selection(image_data, k):
    sorted_data = sorted(image_data, key=lambda x: x['confusion_score'])
    n = len(sorted_data) // k
    selected_images = [item for item in sorted_data[::n]][:k]
    return selected_images

def save_image_data(image_data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['path', 'loss', 'confusion_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in image_data:
            writer.writerow(item)

# Main training loop
for epoch in range(start_epoch, start_epoch+1):
    train(epoch)
    image_data = test(epoch)
    scheduler.step()

    # Save all image data
    save_image_data(image_data, f'image_data_epoch_{epoch}.csv')

    if epoch == 0:  # After the first epoch, select images for downstream task
        selected_images = active_confusion_selection(image_data, args.k)
        save_image_data(selected_images, 'selected_images.csv')
        print(f"Selected {len(selected_images)} images for downstream task")

print("Finished Training")
