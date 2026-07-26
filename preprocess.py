import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COLUMN = "HighSalary"

NUMERIC_FEATURES = [
    "Age",
    "Experience",
    "Education",
    "Hours",
]

CATEGORICAL_FEATURES = [
    "Department",
]

def load_data(csv_path):
    # Read csv
    df = pd.read_csv(csv_path)

    # Separate x and y
    x = df.drop(columns=["Salary", TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split
    xtrain, xtemp, ytrain, ytemp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    xval, xtest, yval, ytest = train_test_split(
        xtemp,
        ytemp,
        test_size=0.5,
        random_state=42,
        stratify=ytemp,
    )

    return (
        xtrain,
        xval,
        xtest,
        ytrain,
        yval,
        ytest,
    )

def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor

def transform_data(
    preprocessor,
    data,
):
    return preprocessor.transform(data)

def save_preprocessor(
    preprocessor,
    preprocessor_path,
):
    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

def load_preprocessor(
    preprocessor_path,
):
    preprocessor = joblib.load(preprocessor_path)

    return preprocessor

