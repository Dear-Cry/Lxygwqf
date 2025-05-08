import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import caltech101 

if __name__ == '__main__':

    device = 'cpu'
    data_dir = 'caltech-101/'
    pretrained = True

    batch_size = 64
    num_classes = 101

    if pretrained : 
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else :
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])

    test_dataset = caltech101.Caltech101(data_dir, split='test', transform=test_transform, train_ratio=0.8, seed=42)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def load_model(network_type='alexnet', num_classes=101):
        if network_type == 'alexnet':
            model = torchvision.models.alexnet(weights=None)
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
        elif network_type == 'vgg':
            model = torchvision.models.vgg16(weights=None)
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
        elif network_type == 'resnet':
            model = torchvision.models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    model = load_model(network_type='resnet')  
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()  

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {correct / total:.4f}')