import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from dataset import create_dataloader
from model import EmployeeClassifier
from preprocess import (
    load_data,
    create_preprocessor,
    transform_data,
    save_preprocessor,
)
from utils import save_checkpoint, load_checkpoint, save_best_model

CSV_PATH = "data/employees.csv"
BEST_MODEL_PATH = "models/best_model.pth"
CHECKPOINT_PATH = "models/checkpoint.pth"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
) = load_data(CSV_PATH)

preprocessor = create_preprocessor()

xtrain = preprocessor.fit_transform(xtrain)

save_preprocessor(
    preprocessor,
    PREPROCESSOR_PATH,
)

xval = transform_data(
    preprocessor,
    xval,
)


train_loader = create_dataloader(
    xtrain,
    ytrain,
    batch_size=32,
    shuffle=True,
)

val_loader = create_dataloader(
    xval,
    yval,
    batch_size=32,
    shuffle=False,
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

scheduler=ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5
)

start_epoch=0
early_stop_counter = 0
best_val_loss = float("inf")
if os.path.exists(CHECKPOINT_PATH):

    (
        last_epoch,
        best_val_loss,
        early_stop_counter,
    ) =load_checkpoint(
        model,
        optimizer,
        scheduler,
        CHECKPOINT_PATH,
        device,
    )
    start_epoch = last_epoch + 1

    print(f"Resuming from Epoch {start_epoch}")


epochs=100
early_stopping_patience = 10
for epoch in range(start_epoch,epochs):
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
    # avg val loss
    val_loss/=len(val_loader)

    #save best model and handle ealry stopping
    if val_loss < best_val_loss:

        best_val_loss=val_loss
        early_stop_counter=0

        save_best_model(
            model,
            BEST_MODEL_PATH,
        )

        print(f"Best model saved at Epoch {epoch+1} (Val Loss: {val_loss:.4f})")

    else:
        early_stop_counter+=1

    #update lr
    scheduler.step(val_loss)

    #checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        best_val_loss,
        early_stop_counter,
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved at Epoch {epoch+1}")

    #current lr
    current_lr=optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"LR: {current_lr:.6f}" 
    ) 

    #stop training if there is no improvemnt
    if early_stop_counter >= early_stopping_patience:
        print("early stopping triggered")
        break

    
