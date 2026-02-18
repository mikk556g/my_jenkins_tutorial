import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.models as models
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from itertools import chain
from torch.utils.data import WeightedRandomSampler


dataset_path = r"/ceph/home/student.aau.dk/rk33gs/my_datasets/miniprojekt_dataset"


classes_to_idx = {
            "angry": 0,
            "disgust": 1, 
            "fear": 2, 
            "happy": 3, 
            "neutral": 4, 
            "sad": 5, 
            "surprise": 6
            }



image_path_list = []
image_label_list = []


for class_name, class_idx in classes_to_idx.items():
    folder_path = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_path_list.append(file_path)
        image_label_list.append(class_idx)



train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(image_path_list, image_label_list, test_size=0.1, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=0.11, random_state=42)


val_test_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = torch.tensor(self.img_labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label



test_set = CustomDataset(test_paths, 
                         test_labels, 
                         transform=val_test_transform)

test_dataloader = DataLoader(test_set, 
                             batch_size=32, 
                             shuffle=False, 
                             num_workers=2, 
                             pin_memory=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")



model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 7)
)

for param in model.fc.parameters():
        param.requires_grad = True

model.load_state_dict(torch.load("best_resnet50_emotion1.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_test, y_test in test_dataloader:
        output = model(X_test.to(device))
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=classes_to_idx.keys()))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (rows = true labels, columns = predicted labels):")
print(np.array2string(cm, separator=', '))
