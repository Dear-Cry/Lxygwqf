import os
import random
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import Subset, DataLoader 

import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50
from torchvision.models import AlexNet_Weights,VGG16_Weights, ResNet18_Weights, ResNet50_Weights

from tqdm import tqdm

import caltech101
import plot

if __name__ == '__main__':
    """Hyperparameters"""

    device = 'cpu' # 'cpu', 'cuda'

    num_classes = 101

    network_type = 'resnet18' # 'alexnet', 'vgg16', 'resnet18', 'resnet50'
    batch_size = 64

    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-5

    num_epochs = 30
    step_size = 20
    gamma = 0.1

    log_frequency = 10
    patience = 5
    pretrained = True
    freeze = 'conv_layers' # 'no_freezing', 'conv_layers'

    train_val_ratio = 0.7

    """Data Preprocessing"""
        
    data_dir = 'caltech-101/'

    if pretrained: 
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
    train_dataset = caltech101.Caltech101(data_dir, split='train', transform=train_transform, train_ratio=0.8, seed=42)
    # test_dataset = caltech101.Caltech101(data_dir, split='test', transform=train_transform, train_ratio=0.8, seed=42)
    # print(f"training set: {len(train_dataset)} distribution: {np.unique(train_dataset.targets, return_counts=True)}")
    # print(f"test set: {len(test_dataset)} distribution: {np.unique(test_dataset.targets, return_counts=True)}")
    # print(f"numclasses: {len(train_dataset.classes)}")

    # class_to_idx = train_dataset.class_idx
    # labels = train_dataset.targets + test_dataset.targets

    X = [s[0] for s in train_dataset.samples]
    y = [s[1] for s in train_dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_val_ratio, random_state=42)
    for train_idx, val_idx in sss.split(X, y):
        train_indexes = train_idx
        val_indexes = val_idx
    val_dataset = Subset(train_dataset, val_indexes)
    train_dataset = Subset(train_dataset, train_indexes)
    print(f'Training set:{len(train_dataset)}')
    print(f'Validation set:{len(val_dataset)}')
    # print(f'Test set:{len(test_dataset)}')

    # from collections import Counter

    # counter = Counter(labels)
    # count_tuples = counter.most_common()
    # idx_to_class = {v:k for k, v in class_to_idx.items()}
    # label_arr = [idx_to_class[idx] for idx, _ in count_tuples]
    # count_arr = [count for _, count in count_tuples]

    # plt.figure(figsize=(15, 10))
    # plt.grid(axis='y',zorder=0)
    # plt.bar(label_arr, count_arr, align='center', zorder=3)
    # plt.xticks(rotation='vertical')
    # plt.title('Class Distribution in Dataset Caltech101')
    # plt.savefig('distribution101.png')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    '''Show some examples(Note: pretrained=False)'''
    # x, y = next(iter(DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, drop_last=True)))
    # out = torchvision.utils.make_grid(x)
    # img = out / 2 + 0.5
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

    """Training"""

    if network_type == 'alexnet':
        if pretrained:
            net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        else:
            net = alexnet(weights=None)
        net.classifier[6] = nn.Linear(4096, num_classes)
    elif network_type == 'vgg16':
        if pretrained:
            net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            net = vgg16(weights=None)
        net.classifier[6] = nn.Linear(4096, num_classes)
    elif network_type == 'resnet18':
        if pretrained:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            net = resnet18(weights=None)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)
    elif network_type == 'resnet50':
        if pretrained:
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            net = resnet50(weights=None)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)
    else:
        raise (ValueError(f"Error Network Type (network_type = {network_type}) \n Possible values are: 'alexnet', 'vgg16', 'resnet18', 'resnet50' "))
    print(net)
    if network_type in ['alexnet', 'vgg16']:
        if freeze == 'no_freezing':
            parameters_to_optimize = net.parameters()
        elif freeze == 'conv_layers':
            for param in net.parameters():
                param.requires_grad = False
            for param in net.classifier.parameters():
                param.requires_grad = True
            parameters_to_optimize = net.classifier.parameters()
        else:
            raise (ValueError(f"Error Freezing Layers (freeze = {freeze}) \n Possible values are: 'no_freezing', 'conv_layers' "))
    elif network_type in ['resnet18', 'resnet50']:
        if freeze == 'no_freezing':
            parameters_to_optimize = net.parameters()
        elif freeze == 'conv_layers':
            for param in net.parameters():
                param.requires_grad = False
            for param in net.fc.parameters():
                param.requires_grad = True
            parameters_to_optimize = net.fc.parameters()
        else:
            raise (ValueError(f"Error Freezing Layers (freeze = {freeze}) \n Possible values are: 'no_freezing', 'conv_layers' "))
    else:
        raise (ValueError(f"Error Network Type (network_type = {network_type}) \n Possible values are: 'alexnet', 'vgg16', 'resnet18', 'resnet50' "))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.Adam(parameters_to_optimize, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # EVAL_ACCURACY_ON_TRAINING = True
    # criterion_val = nn.CrossEntropyLoss(reduction='sum')

    start = time.time()
    net = net.to(device)
    cudnn.benchmark 

    best_net = 0
    best_epoch = 0
    best_val_acc = 0.0
    no_improve_epochs = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}, LR = {scheduler.get_last_lr()}")

        net.train()

        running_acc_train = 0
        running_loss_train = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output_train = net(images)
                _, preds = torch.max(output_train, dim=1)
                loss = criterion(output_train, labels)

                loss.backward()
                optimizer.step()

            running_acc_train += torch.sum(preds == labels).item()
            running_loss_train += loss.item() * images.size(0)
        
        train_acc = running_acc_train / float(len(train_dataset))
        train_loss = running_loss_train / float(len(train_dataset))
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        net.train(False)

        running_acc_val = 0.0
        running_loss_val = 0.0

        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                output_val = net(images)
                _, preds = torch.max(output_val, dim=1)
                loss = criterion(output_val, labels)
            
            running_acc_val += torch.sum(preds == labels).item()
            running_loss_val += loss.item() * images.size(0)

        val_loss = running_loss_val / float(len(val_dataset))
        val_acc = running_acc_val / float(len(val_dataset))
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        epoch += 1
        if epoch % log_frequency == 0:
            print(f'Epoch:{epoch}, Training Loss:{train_loss}, Train Acc:{train_acc}, Validation Loss:{val_loss}, Validation Acc:{val_acc}')
        if val_acc > best_val_acc:
            print(f'Best model has been saved: {best_val_acc}->{val_acc}')
            no_improve_epochs = 0
            best_val_acc = val_acc
            best_epoch = epoch
            best_net = copy.deepcopy(net)
        # else:
        #     no_improve_epochs += 1
        #     if no_improve_epochs > patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break

        scheduler.step()

    print(f"\nBest epoch: {best_epoch+1}\n{best_val_acc:.4f} (Validation Accuracy)\n")
    print(f"> In {(time.time()-start)/60:.2f} minutes")

    plot.plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)

    torch.save(best_net.state_dict(), 'best_model.pth')
    print("Saved best model to 'best_model.pth'")
        
    # best_net = best_net.to(device)
    # best_net.train(False)

    # running_acc_test = 0.0
    # for images, labels in tqdm(test_dataloader):
    #     images.to(device)
    #     labels.to(device)

    #     output_test = best_net(images)
    #     _, preds = torch.max(output_test, dim=1)
    #     running_acc_test += torch.sum(preds == labels).item()

    # test_acc = running_acc_test / float(len(test_dataset))

    # print(f'Test Accuracy: {test_acc}')