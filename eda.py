import pandas as pd

def explore_data(csv_path):
    df = pd.read_csv(csv_path)

    print("=" * 50)
    print("First 5 Rows")
    print("=" * 50)
    print(df.head())

    print("\n" + "=" * 50)
    print("Dataset Shape")
    print("=" * 50)
    print(df.shape)

    print("\n" + "=" * 50)
    print("Column Names")
    print("=" * 50)
    print(df.columns.tolist())

    print("\n" + "=" * 50)
    print("Data Types")
    print("=" * 50)
    print(df.dtypes)

    print("\n" + "=" * 50)
    print("Missing Values")
    print("=" * 50)
    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("Target Distribution")
    print("=" * 50)
    print(df["HighSalary"].value_counts())

    print("\nPercentage")
    print(df["HighSalary"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    explore_data("data/employees.csv")