import numpy as np

def normalize_1d(data: np.ndarray) -> np.ndarray:
    """
    Normalizes a 1D array using z-score normalization (subtract mean, divide by standard deviation).

    Args:
        data (np.ndarray): Input 1D array to normalize.

    Returns:
        np.ndarray: Normalized 1D array.
    """
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data)
    std = np.std(data) if np.std(data) != 0 else 1.0
    return (data - mean) / std

def normalize_2d(data: np.ndarray) -> np.ndarray:
    """
    Normalizes each column of a 2D array using z-score normalization.

    Args:
        data (np.ndarray): Input 2D array to normalize (n_samples, n_features).

    Returns:
        np.ndarray: Normalized 2D array.
    """
    data = np.asarray(data, dtype=np.float64)
    return np.apply_along_axis(normalize_1d, axis=0, arr=data)

def create_split(X, y, seed=42):
    """
    Splits data into training and testing sets with an 80:20 ratio.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target array of shape (n_samples,).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training targets.
            - X_test (np.ndarray): Testing features.
            - y_test (np.ndarray): Testing targets.
    """
    # Ensure y is 2D column vector
    y = y.reshape(-1, 1)
    
    # Calculate split point (80% for training)
    data_len = len(X)
    train_size = round(0.8 * data_len)
    
    # Shuffle indices
    np.random.seed(seed)
    indices = np.random.permutation(data_len)
    
    # Split data
    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_test = X[indices[train_size:]]
    y_test = y[indices[train_size:]]
    
    return X_train, y_train, X_test, y_test