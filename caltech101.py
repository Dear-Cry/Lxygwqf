import os
import random
from PIL import Image
from torchvision.datasets.vision import VisionDataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
  
class Caltech101(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, train_ratio=0.8, seed=42):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        random.seed(seed)
        self.classes, self.class_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.loader = pil_loader
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self):
        root = os.path.join(self.root, '101_ObjectCategories')
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.remove('BACKGROUND_Google')
        classes.sort()
        class_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_idx
    
    def _make_dataset(self):
        root = os.path.join(self.root, '101_ObjectCategories')
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        
        random.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)
        if self.split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target 
    
    def __len__(self):
        return len(self.samples)