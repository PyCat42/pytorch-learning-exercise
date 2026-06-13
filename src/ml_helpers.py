import torch

def train_test_split(X_data, y_data, train_size=0.8):
    """
    Split data into train and test sets.
    :param X_data: data
    :param y_data: labels
    :param train_size:
    :return: X_train, y_train, X_test, y_test
    """
    num_samples = len(X_data)

    # Shuffle indices for randomness
    indices = torch.randperm(num_samples)

    # Calculate how much data belong to training sample
    train_split = int(train_size * len(X_data))

    train_indices = indices[:train_split]
    test_indices = indices[train_split:]

    X_train, y_train = X_data[train_indices], y_data[train_indices]
    X_test, y_test = X_data[test_indices], y_data[test_indices]

    return X_train, y_train, X_test, y_test
