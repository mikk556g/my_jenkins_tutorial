import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FineTuned(nn.Module):
    def __init__(self, model_config):
        super(ResNet50FineTuned, self).__init__()

        num_classes = model_config['num_classes']
        pretrained = model_config.get('pretrained', True)
        freeze_layers = model_config.get('freeze_layers', ["layer1", "layer2", "layer3"])
        dropout_fc1 = model_config.get('dropout_fc1', 0.4)
        dropout_fc2 = model_config.get('dropout_fc2', 0.3)
        hidden_units = model_config.get('hidden_units', 256)

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Freeze specified layers
        layers_dict = dict(self.backbone.named_children())
        for layer_name in freeze_layers:
            if layer_name in layers_dict:
                for param in layers_dict[layer_name].parameters():
                    param.requires_grad = False

        # Replace fully connected head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_fc1),
            nn.Linear(self.backbone.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_fc2),
            nn.Linear(hidden_units, num_classes)
        )

        # Ensure FC parameters are trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
