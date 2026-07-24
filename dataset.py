import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloaders(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
    batch_size=32,
):
    xtrain = torch.tensor(xtrain, dtype=torch.float32)
    xval = torch.tensor(xval, dtype=torch.float32)
    xtest = torch.tensor(xtest, dtype=torch.float32)

    ytrain = torch.tensor(ytrain.values, dtype=torch.float32).unsqueeze(1)
    yval = torch.tensor(yval.values, dtype=torch.float32).unsqueeze(1)
    ytest = torch.tensor(ytest.values, dtype=torch.float32)

    train_dataset = TensorDataset(xtrain, ytrain)
    val_dataset = TensorDataset(xval, yval)
    test_dataset = TensorDataset(xtest, ytest)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader