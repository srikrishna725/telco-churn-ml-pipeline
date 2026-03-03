import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps:
    - Convert TotalCharges to numeric
    - Drop missing values
    - Encode target variable (Churn)
    - Drop customerID column

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """

    # Convert TotalCharges to numeric (coerce invalid values to NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID (not useful for modeling)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df