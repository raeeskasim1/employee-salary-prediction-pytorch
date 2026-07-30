import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


from preprocess import (
    load_data,
    load_preprocessor,
    transform_data,
)

from dataset import create_dataloader
from model import EmployeeClassifier
from utils import load_best_model

CSV_PATH = "data/employees.csv"
BEST_MODEL_PATH = "models/best_model.pth"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

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
) = load_data(CSV_PATH)

preprocessor = load_preprocessor(
    PREPROCESSOR_PATH,
)

xtest = transform_data(
    preprocessor,
    xtest,
)

test_loader = create_dataloader(
    xtest,
    ytest,
    batch_size=32,
    shuffle=False,
)

input_size = xtest.shape[1]

model = EmployeeClassifier(input_size)
model.to(device)

load_best_model(model,BEST_MODEL_PATH,device)

model.eval()

loss_fn = nn.BCELoss()

test_loss = 0
all_predictions=[]
all_labels=[]

with torch.no_grad():
    for batch_x,batch_y in test_loader:
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        prediction=model(batch_x)
        loss=loss_fn(prediction,batch_y)
        test_loss+=loss.item()

        predicted=(prediction >= 0.5).float()

        all_predictions.extend(predicted.cpu().numpy().flatten())
        all_labels.extend(batch_y.cpu().numpy().flatten())

test_loss/=len(test_loader)
accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
)

recall = recall_score(
    all_labels,
    all_predictions,
)

f1 = f1_score(
    all_labels,
    all_predictions,
)

conf_matrix = confusion_matrix(
    all_labels,
    all_predictions,
)  

print(f"Test Loss : {test_loss:.4f}")
print(f"Accuracy  : {accuracy:.4%}")
print(f"Precision : {precision:.4%}")
print(f"Recall    : {recall:.4%}")
print(f"F1 Score  : {f1:.4%}")

print("\nConfusion Matrix")
print(conf_matrix)

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["Low Salary", "High Salary"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig(
    "assets/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()