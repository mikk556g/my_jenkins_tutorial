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


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness= 0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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



train_set = CustomDataset(train_paths,
                          train_labels,
                          transform=train_transform)

val_set = CustomDataset(val_paths,
                        val_labels,
                        transform=val_test_transform)

test_set = CustomDataset(test_paths,
                         test_labels,
                         transform=val_test_transform)

class_counts = Counter(train_labels)
num_classes = len(classes_to_idx)
total_samples = len(train_labels)
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]


train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


model = models.resnet50(weights="IMAGENET1K_V1")

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

model = model.to(device)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-4}], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=1e-4,
                                                epochs=15,
                                                steps_per_epoch=len(train_dataloader),
                                                pct_start=0.1,
                                                anneal_strategy="cos")

train_loss_list = []
val_loss_list = []

epochs = 15
best_acc = 0.0
for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")
    running_train_loss = 0.0
    running_train_corrects = 0.0


    model.train()
    for X_train, y_train in train_dataloader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()

        output_train = model(X_train)
        train_loss = criterion(output_train, y_train)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        _, preds = torch.max(output_train, 1)
        running_train_corrects += torch.sum(preds == y_train.data).item()
        running_train_loss += train_loss.item()

    train_epoch_acc = running_train_corrects / len(train_set)
    train_epoch_loss = running_train_loss / len(train_dataloader)
    train_loss_list.append(train_epoch_loss)
    print(f"Training loss: {train_epoch_loss:.4f} Training accuracy: {train_epoch_acc:.4f}")


    model.eval()
    running_val_loss = 0.0
    running_val_corrects = 0.0
    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            output_val = model(X_val)
            val_loss = criterion(output_val, y_val)
            running_val_loss += val_loss.item()

            _, preds = torch.max(output_val, 1)
            running_val_corrects += torch.sum(preds == y_val.data).item()

    val_epoch_acc = running_val_corrects / len(val_set)
    val_epoch_loss = running_val_loss / len(val_dataloader)
    val_loss_list.append(val_epoch_loss)

    print(f"Validation loss: {val_epoch_loss:.4f} Validation accuracy: {val_epoch_acc:.4f}")

    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, "best_resnet50_emotion1.pth")

print(f"Best evaluation accuracy", best_acc)
model.load_state_dict(best_model_state)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_val, y_val in val_dataloader:
        output = model(X_val.to(device))
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_val.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=classes_to_idx.keys()))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (rows = true labels, columns = predicted labels):")
print(np.array2string(cm, separator=', '))

plt_epochs = range(1, epochs + 1)
plt.plot(plt_epochs, train_loss_list, label='Train Loss')
plt.plot(plt_epochs, val_loss_list, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()
