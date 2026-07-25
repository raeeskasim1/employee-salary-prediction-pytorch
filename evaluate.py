import torch
import torch.nn as nn

from preprocess import load_and_preprocess_data
from dataset import create_dataloaders
from model import EmployeeClassifier

CSV_PATH = "data/employees.csv"
BEST_MODEL_PATH = "models/best_model.pth"

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
    preprocessor,
) = load_and_preprocess_data(CSV_PATH)

train_loader, val_loader, test_loader = create_dataloaders(
    xtrain,
    xval,
    xtest,
    ytrain,
    yval,
    ytest,
)

input_size = xtrain.shape[1]

model = EmployeeClassifier(input_size)
model.to(device)

model.load_state_dict(
    torch.load(BEST_MODEL_PATH, map_location=device)
)

model.eval()

loss_fn = nn.BCELoss()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch_x,batch_y in test_loader:
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        prediction=model(batch_x)
        loss=loss_fn(prediction,batch_y)
        test_loss+=loss.item()

        predicted=(prediction >= 0.5).float()
        correct+=(predicted == batch_y).sum().item()
        total+=batch_y.size(0)
test_loss/=len(test_loader)
accuracy=correct / total   

print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4%}")
