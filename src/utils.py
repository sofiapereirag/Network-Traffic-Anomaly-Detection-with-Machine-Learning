# This file is for the utility functions used in the project.
import pandas as pd
import numpy as np
import joblib

def load_data_file(file_path):
    """
    Load the dataset from a single CSV file.
    Parameters:
    - file_path: str, path to the CSV file containing the dataset.
    Returns:
    - df: DataFrame.
    """
    """ n_total = sum(1 for _ in open(file_path)) - 1   # total number of rows without header
    n_total = np.arange(1, n_total+1)               # create an array of all row indices
    # Randomly select rows to skip (just to read 50000 rows)
    rows_to_skip = sorted(np.random.choice(n_total, n_total.size - 50000, replace=False))
    df = pd.read_csv(file_path, skiprows=rows_to_skip) """
    df = pd.read_csv(file_path)
    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Add binary column for attack vs. normal traffic
    if 'Label' in df.columns:
        # Remove duplicated header rows
        df = df[df['Label'] != 'Label']
        df['Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    else:
        print(f"Warning: 'Label' column not found in {file_path}.")
    
    return df

def load_data_files(file_paths):
    """
    Load the dataset from multiple CSV files.
    Only loads the first 50,000 rows of each file.
    Parameters:
    - file_paths: str, path to the CSV file containing the dataset.
    Returns:
    - df: DataFrame.
    """
    df_list = []
    for path in file_paths:
        n_total = sum(1 for _ in open(path)) - 1   # total number of rows without header
        n_total = np.arange(1, n_total+1)               # create an array of all row indices
        # Randomly select rows to skip (just to read 50000 rows)
        rows_to_skip = sorted(np.random.choice(n_total, n_total.size - 50000, replace=False))
        df = pd.read_csv(path, skiprows=rows_to_skip)

        # Remove spaces from column names
        df.columns = df.columns.str.strip()

        # Add binary column for attack vs. normal traffic
        if 'Label' in df.columns:
            # Remove duplicated header rows
            df = df[df['Label'] != 'Label']
            df['Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        else:
            print(f"Warning: 'Label' column not found in {path}.")
    
        df_list.append(df)

    df_list = pd.concat(df_list, ignore_index=True)
    return df_list

def display_data_info(df):
    """ 
    Display basic information about the dataset.
    Parameters:
    - df: DataFrame, the dataset to display information for.
    """
    print("=== NUMBER OF ROWS ===")
    print(len(df))

    print("\n=== DATA TYPES OF EACH COLUMN ===")
    print(df.dtypes)

    print("\n=== ALL COLUMN NAMES ===")
    print(df.columns.tolist())

    # Verify the new attack column
    print("\n ' Attack' column created with value counts:", df["Attack"].value_counts())

    # Plot the distribution of traffic classes
    df['Attack'].value_counts().plot(kind='bar', title='Distribution of Traffic Classes')

def display_distribution(df):
    """
    Display the distribution of the target variable.
    Parameters:
    - df: DataFrame, the dataset to display the distribution for.
    """
    # Verify the new attack column
    print("\n ' Attack' column created with value counts:", df["Attack"].value_counts())

def save_object(object, object_path):
    """
    Save the object to a file.
    Parameters: 
    - object: The object to save.
    - object_path: str, path to save the object.
    """
    joblib.dump(object, object_path)
    print(f"Object saved to {object_path}.")

def load_saved_object(object_path):
    """
    Load a saved object from a file.
    Parameters:
    - object_path: str, path to the saved object.
    Returns:
    - object: The loaded object.
    """
    return joblib.load(object_path)

