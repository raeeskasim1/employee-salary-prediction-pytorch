import random
import numpy as np
import torch



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    early_stop_counter,
    checkpoint_path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop_counter,
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    early_stop_counter = checkpoint["early_stop_counter"]

    return (
        epoch,
        best_val_loss,
        early_stop_counter,
    )

def save_best_model(
    model,
    best_model_path,
):
    torch.save(
        model.state_dict(),
        best_model_path,
    )

def load_best_model(model,best_model_path,device):
    model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)