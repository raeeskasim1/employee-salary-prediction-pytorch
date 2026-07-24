import pandas as pd

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

def load_and_preprocess_data(csv_path):
    
    """
    Load the employee dataset, split it into train/validation/test sets,
    preprocess the features, and return the processed data along with the
    fitted preprocessor.
    """

    #--------Read csv-------
    df=pd.read_csv(csv_path)

    #-----separate x and y-----
    x=df.drop(columns=['Salary',TARGET_COLUMN])
    y=df[TARGET_COLUMN]

    #-------split--------
    xtrain,xtemp,ytrain,ytemp=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
    xval,xtest,yval,ytest=train_test_split(xtemp,ytemp,test_size=0.5,random_state=42,stratify=ytemp)

    #-----pipeline-----
    preprocessor=ColumnTransformer(
        transformers=[
            (
                'num',StandardScaler(),NUMERIC_FEATURES
                ),
            (
                'cat',OneHotEncoder(handle_unknown="ignore"),CATEGORICAL_FEATURES
            )
        ]
    )

    xtrain=preprocessor.fit_transform(xtrain)
    xval=preprocessor.transform(xval)
    xtest=preprocessor.transform(xtest)

    return (
        xtrain,
        xval,
        xtest,
        ytrain,
        yval,
        ytest,
        preprocessor,
    )





