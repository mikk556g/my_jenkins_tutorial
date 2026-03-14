from torch.optim.lr_scheduler import OneCycleLR

def onecyclelr(optimizer, scheduler_config, train_dataloader):

    steps_per_epoch=len(train_dataloader)

    if scheduler_config["name"].lower() == "onecyclelr":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_config['max_lr'],
            epochs=scheduler_config['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=scheduler_config['pct_start'],
            anneal_strategy=scheduler_config['anneal_strategy']
        )
    else:
        raise ValueError("Unsupported scheduler")

    return scheduler
