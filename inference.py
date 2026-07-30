import torch
import pandas as pd

from model import EmployeeClassifier
from utils import load_best_model
from preprocess import (
    load_preprocessor,
    transform_data,
)

CSV_PATH = "data/employees.csv"
BEST_MODEL_PATH = "models/best_model.pth"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

preprocessor = load_preprocessor(
    PREPROCESSOR_PATH,
)

# new_employee = pd.DataFrame({
#     "Age": [26],
#     "Experience": [0],
#     "Education": [18],
#     "Hours": [45],
#     "Department": ["IT"]
# })

new_employee = pd.DataFrame({
    "Age": [30],
    "Experience": [7],
    "Education": [16],
    "Hours": [25],
    "Department": ["Management"]
})

new_employee = transform_data(
    preprocessor,
    new_employee,
)

new_employee = torch.tensor(
    new_employee,
    dtype=torch.float32
).to(device)

input_size = new_employee.shape[1]

model = EmployeeClassifier(input_size)
model.to(device)

load_best_model(model,BEST_MODEL_PATH,device)

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