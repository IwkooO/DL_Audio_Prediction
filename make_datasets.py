import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

def make_dataset_balanced(padded_arrays, valence_values, batch_size=256):

    X_train, X_test_help, y_train, y_test_help = train_test_split(padded_arrays, valence_values, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test_help, y_test_help, test_size=0.5, random_state=42)
    
    # Convert input data and labels to tensors

    bin_edges = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]

    # Transform continuous target values into categorical labels
    y_train_categorical = np.digitize(y_train, bins=bin_edges)

    # Convert to integer type
    y_train_categorical = y_train_categorical.astype(int)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_categorical, dtype=torch.int)

    # Create a dataset from tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Apply Random Oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train_tensor, y_train_tensor)

    # Convert oversampled data back to PyTorch tensors
    X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_resampled_tensor = torch.tensor(y_resampled, dtype=torch.float32)

    # Create a dataset from resampled tensors
    resampled_dataset = TensorDataset(X_resampled_tensor, y_resampled_tensor)

    # Create a DataLoader
    resampled_loader = DataLoader(resampled_dataset, batch_size=batch_size, shuffle=True)

    # Similarly, create data loaders for test and validation sets
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X_validation_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_val, dtype=torch.float32)
    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return resampled_loader, test_loader, validation_loader


def make_dataset(padded_arrays, valence_values, batch_size=256):

    
    X_train, X_test_help, y_train, y_test_help = train_test_split(padded_arrays, valence_values, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test_help, y_test_help, test_size=0.5, random_state=42)

    # Convert input data and labels to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Use float32 for input features
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Use float32 for labels

    # Create a dataset from tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Repeat the same process for the test set

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Repeat the same process for the validation set
    X_validation_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_val, dtype=torch.float32)

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, validation_loader