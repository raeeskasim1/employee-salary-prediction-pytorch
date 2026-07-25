import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    checkpoint_path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, checkpoint_path)