# Employee Salary Prediction using PyTorch

## 📌 Overview

This project is an end-to-end **binary classification** machine learning application built with **PyTorch**. The model predicts whether an employee belongs to the **High Salary** category based on demographic and work-related features.

The project follows a clean and modular architecture with separate components for data preprocessing, model definition, training, evaluation, and inference.

---

##  Features

* End-to-end PyTorch training pipeline
* Data preprocessing using `ColumnTransformer`
* Feature scaling with `StandardScaler`
* Categorical encoding using `OneHotEncoder`
* Train/Validation/Test split
* Mini-batch training with `DataLoader`
* Neural Network built using `nn.Sequential`
* GPU support (CUDA when available)
* Learning Rate Scheduler (`ReduceLROnPlateau`)
* Early Stopping
* Checkpoint Saving & Resume Training
* Best Model Saving
* Model Evaluation
* Inference on new employee data

---

## 📂 Project Structure

```text
employee/
│
├── assets/
│   └── (images/screenshots/plots)
│
├── data/
│   └── employees.csv
│
├── models/
│   ├── best_model.pth
│   ├── checkpoint.pth
│   └── preprocessor.pkl
│
├── dataset.py
├── preprocess.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains employee information used to predict salary category.

### Features

* Age
* Experience
* Education
* Hours
* Department

### Target

* `HighSalary`

  * `1` → High Salary
  * `0` → Low Salary

**Note:** The original `Salary` column is removed during preprocessing because it would leak information about the target.

---

## Preprocessing

The preprocessing pipeline is implemented using **Scikit-learn's `ColumnTransformer`**.

### Numerical Features

* Age
* Experience
* Education
* Hours

Applied Transformation:

* StandardScaler

### Categorical Features

* Department

Applied Transformation:

* OneHotEncoder (`handle_unknown="ignore"`)

The fitted preprocessor is saved as:

```text
models/preprocessor.pkl
```

and reused during evaluation and inference to ensure consistent preprocessing.

---

## Model Architecture

The model is a fully connected neural network implemented with PyTorch.

```text
Input
   │
Linear (Input → 16)
   │
ReLU
   │
Linear (16 → 8)
   │
ReLU
   │
Linear (8 → 1)
   │
Sigmoid
```

Loss Function:

* Binary Cross Entropy Loss (`BCELoss`)

Optimizer:

* Adam

Learning Rate Scheduler:

* ReduceLROnPlateau

---

## Training

The training pipeline includes:

* Forward pass
* Loss computation
* Backpropagation
* Optimizer update
* Validation after each epoch
* Learning rate scheduling
* Early stopping
* Checkpoint saving
* Best model saving

Saved files:

```text
models/
├── best_model.pth
├── checkpoint.pth
└── preprocessor.pkl
```

---

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example results:

```text
Test Loss : 0.0780
Accuracy  : 96.5333%
Precision : 96.6942%
Recall    : 97.9079%
F1 Score  : 97.2973%

Confusion Matrix
[[512  32]
 [ 20 936]]
```

---

# Training Visualization

Training curves and evaluation visualizations are available in the `assets/` folder.

Examples:
- Training and validation loss curves
- Model performance metrics  
- Confusion matrix visualization  

## Inference

A new employee can be predicted by modifying the sample in `inference.py`.

Example:

```python
new_employee = pd.DataFrame({
    "Age": [45],
    "Experience": [20],
    "Education": [18],
    "Hours": [50],
    "Department": ["Management"]
})
```

Example output:

```text
Probability : 99.99%
Prediction  : High Salary
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd employee
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Usage

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

### Run inference

```bash
python inference.py
```

---

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## What I Learned

Through this project, I implemented:

* Modular project structure
* Data preprocessing pipelines
* Neural network implementation in PyTorch
* Training and validation workflows
* Model checkpointing
* Resume training
* Learning rate scheduling
* Early stopping
* Model evaluation
* Inference on unseen data

---

## Future Improvements

Possible enhancements include:

* Hyperparameter tuning
* Cross-validation
* Experiment tracking
* TensorBoard integration
* Model export with TorchScript or ONNX
* Deployment using FastAPI or Flask
* Docker containerization
