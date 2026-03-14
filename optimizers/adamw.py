import torch.optim as optim

def adamw(model, model_config, optimizer_config):
    
    freeze_layers = model_config["freeze_layers"]
    
    for name, module in model.named_children():
        if name in freeze_layers:
            for param in module.parameters():
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_config["name"].lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    else:
        raise ValueError("Unsupported optimizer")

    return optimizer
