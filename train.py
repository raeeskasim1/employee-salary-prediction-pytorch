import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import load_and_preprocess_data
from dataset import create_dataloaders
from model import EmployeeClassifier

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
    preprocessor,
) = load_and_preprocess_data("data/employees.csv")


#create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
)

#create model
input_size=xtrain.shape[1]
model=EmployeeClassifier(input_size)
model.to(device)


loss_fn=nn.BCELoss()

optimizer=optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs=100
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss=0
    for batch_x,batch_y in train_loader:
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        prediction=model(batch_x)
        loss=loss_fn(prediction,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    train_loss/=len(train_loader)

    #---------validation--------
    model.eval()
    val_loss=0
    with torch.no_grad():
        for batch_x,batch_y in val_loader:
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            prediction=model(batch_x)
            loss=loss_fn(prediction,batch_y)
            val_loss+=loss.item()
    val_loss/=len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    ) 

    if val_loss < best_val_loss:
        best_val_loss=val_loss

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

    print("best model saved")
