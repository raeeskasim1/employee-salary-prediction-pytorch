import torch
import pandas as pd

from preprocess import load_and_preprocess_data
from model import EmployeeClassifier

CSV_PATH = "data/employees.csv"
BEST_MODEL_PATH = "models/best_model.pth"

device = torch.device(
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

new_employee = pd.DataFrame({
    "Age": [26],
    "Experience": [0],
    "Education": [18],
    "Hours": [45],
    "Department": ["IT"]
})

new_employee=preprocessor.transform(new_employee)

new_employee = torch.tensor(
    new_employee,
    dtype=torch.float32
).to(device)

input_size = xtrain.shape[1]

model = EmployeeClassifier(input_size)
model.to(device)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    prediction = model(new_employee)

probability = prediction.item()

if probability >= 0.5:
    predicted_class = "High Salary"
else:
    predicted_class = "Low Salary"

print(f"Probability : {probability:.2%}")
print(f"Prediction  : {predicted_class}")