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
import seaborn as sea
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from itertools import chain
from torch.utils.data import WeightedRandomSampler
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

mlflow.set_experiment("resnet50-emotion-classifier")

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


with mlflow.start_run(run_name="evaluation"):
    
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(
        "resnet50-emotion-classifier", stages=["Staging"])[0]
    model = mlflow.pytorch.load_model(f"models:/resnet50-emotion-classifier/{latest_version_info.version}")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            output = model(X_test)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    report = classification_report(
            all_labels,
            all_preds,
            target_names=classes_to_idx.keys(),
            output_dict=True)

    accuracy = report["accuracy"]

    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Test Accuracy: {accuracy}")

    if accuracy < 0.70:
        raise ValueError("Model performance below threshold!")
    else:

        # Promote it to Production and archive existing Production versions
        client.transition_model_version_stage(
            name="resnet50-emotion-classifier",
            version=latest_version_info.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {latest_version_info} promoted to Production.")


    print(classification_report(
        all_labels,
        all_preds,
        target_names=classes_to_idx.keys()))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sea.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes_to_idx.keys(),
                yticklabels=classes_to_idx.keys())

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")
