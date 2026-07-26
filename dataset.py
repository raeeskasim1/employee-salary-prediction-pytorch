import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(
    x,
    y,
    batch_size=32,
    shuffle=False,
):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(x, y)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
