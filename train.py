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
import mlflow
import mlflow.pytorch
import time
import yaml
from models.resnet50 import ResNet50FineTuned
from optimizers.adamw import adamw
from schedulers.onecyclelr import onecyclelr

mlflow.set_experiment("resnet50-emotion-classifier")

with open("train_config.yaml") as f:
    config = yaml.safe_load(f)

dataset_config = config['dataset']

model_config = config['model']

train_config = config['train']

optimizer_config = config['optimizer']

scheduler_config = config['scheduler']

evaluate_config = config['evaluate']


dataset_path = dataset_config['dataset_path']


classes_to_idx = config['classes']



image_path_list = []
image_label_list = []


for class_name, class_idx in classes_to_idx.items():
    folder_path = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_path_list.append(file_path)
        image_label_list.append(class_idx)



train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(image_path_list, image_label_list, test_size=dataset_config['test_size'], random_state=dataset_config['random_state'])
train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels, test_size=dataset_config['test_size'], random_state=dataset_config['random_state'])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=train_config['RandomResizedCrop']['size'], 
                                 scale=train_config['RandomResizedCrop']['scale']),
    transforms.RandomHorizontalFlip(p=train_config['RandomHorizontalFlip']['p']),
    transforms.RandomRotation(degrees=train_config['RandomRotation']['degrees']),
    transforms.ColorJitter(brightness=train_config['ColorJitter']['brightness'], 
                           contrast=train_config['ColorJitter']['contrast']),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_config['Normalize']['mean'], 
                         std=train_config['Normalize']['std'])
    ])

val_test_transform = transforms.Compose([
    transforms.Resize(size=evaluate_config['Resize']['size']),
    transforms.CenterCrop(size=evaluate_config['CenterCrop']['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=evaluate_config['Normalize']['mean'], 
                         std=evaluate_config['Normalize']['std'])
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


class_counts = Counter(train_labels)
num_classes = len(classes_to_idx)
total_samples = len(train_labels)
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]



train_dataloader = DataLoader(train_set, batch_size=train_config['batch_size'], 
                              shuffle=train_config['shuffle'], 
                              num_workers=train_config['num_workers'], 
                              pin_memory=train_config['pin_memory'])

val_dataloader = DataLoader(val_set, batch_size=train_config['batch_size'], 
                            shuffle=evaluate_config['shuffle'], 
                            num_workers=train_config['num_workers'], 
                            pin_memory=evaluate_config['pin_memory'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = ResNet50FineTuned(model_config)

model = model.to(device)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = adamw(model, model_config, optimizer_config)

scheduler = onecyclelr(optimizer, scheduler_config, train_dataloader)

train_loss_list = []
val_loss_list = []

epochs = train_config['epochs']

with mlflow.start_run():

    start_time = time.time()

    mlflow.log_param("model", "resnet50")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 1e-4)
            
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

    print(f"Best evaluation accuracy", best_acc)

    mlflow.log_metric("best_val_accuracy", best_acc)

    model.load_state_dict(best_model_state)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            output = model(X_val)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes_to_idx.keys(),
        output_dict=True)

    accuracy = report["accuracy"]

    mlflow.log_metric("final_val_accuracy", accuracy)

    print(classification_report(
        all_labels,
        all_preds,
        target_names=classes_to_idx.keys()))

    plt_epochs = range(1, epochs + 1)
    plt.plot(plt_epochs, train_loss_list, label='Train Loss')
    plt.plot(plt_epochs, val_loss_list, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    mlflow.log_artifact("loss_plot.png")

    end_time = time.time()
    training_duration = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)
    print(f"Training completed in {training_duration:.2f} seconds")

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="resnet50-emotion-classifier")

    # Transition the newly registered model to Staging automatically
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    latest_version_info = client.get_latest_versions("resnet50-emotion-classifier", stages=["None"])[0]
    client.transition_model_version_stage(
        name="resnet50-emotion-classifier",
        version=latest_version_info.version,
        stage="Staging",
        archive_existing_versions=False  # optional
    )
    print(f"Model version {latest_version_info.version} moved to Staging.")
