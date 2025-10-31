import numpy as np
import h5py
from typing import Union

from sklearn.preprocessing import OneHotEncoder


def activity_func_generator(ka_mat, ka_contrib):
    """
    Generates a function that computes the activity based on the King-Altman model.
    The function takes a binary selector vector and a parameter vector.
    """

    def activity_func(s, theta):
        log_k = construct_log_k(s, theta)
        ka_terms = np.exp(np.dot(ka_mat, log_k))
        ka_contrib_terms = np.dot(ka_contrib, ka_terms) / np.sum(ka_terms)

        activity = np.dot(ka_contrib_terms, np.exp(log_k))
        return activity

    return activity_func


def construct_log_k(x, theta):
    s = x[:, 1]  # Extract binary selector from one-hot encoded vector
    a = theta[::2]
    b = theta[1::2]
    return (1 - s) * a + s * b


# ---- Step 3: Loss function for regression ----
def min_mse_loss(theta, f, X, Y):
    total_loss = 0.0
    for x_data, y_data in zip(X, Y):
        y_pred = f(x_data, theta)
        total_loss += (y_pred - y_data) ** 2
    return total_loss


def min_mse_loss_reg(theta, f, X, Y, epsilon=1e-8):
    total_loss = 0.0
    for x_data, y_data in zip(X, Y):
        y_pred = f(x_data, theta)
        total_loss += (y_pred - y_data) ** 2
        total_loss += epsilon * np.log(np.sum(np.exp(theta)))  # Regularization term
    return total_loss


# Extend x_train with random [0,1] or [1,0] vectors
def extend_sequences(sequences, num_extensions=1, random_state=None):
    """
    Extend sequences with random [0,1] or [1,0] vectors

    Args:
        sequences: Input array of shape (n_samples, seq_length, 2)
        num_extensions: Number of random vectors to add
        random_state: Random seed for reproducibility

    Returns:
        Extended sequences of shape (n_samples, seq_length + num_extensions, 2)
    """
    if num_extensions <= 0:
        raise ValueError("num_extensions must be a positive integer")
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, seq_length, n_features = sequences.shape

    # Create random binary choices: 0 for [1,0], 1 for [0,1]
    random_choices = np.random.randint(0, 2, size=(n_samples, num_extensions))

    # Create the extension vectors
    extensions = np.zeros((n_samples, num_extensions, 2))
    extensions[random_choices == 0, 0] = 1  # [1,0] vectors
    extensions[random_choices == 1, 1] = 1  # [0,1] vectors

    # Concatenate along the sequence dimension
    extended_sequences = np.concatenate([sequences, extensions], axis=1)

    return extended_sequences


def load_kinetic_data(
    filepath: str,
    template_seq: Union[str, None] = "AAAAAAA",
    mutant_seq: Union[str, None] = "BBBBBBB",
):
    """Load and preprocess kinetic model data from HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file containing kinetic model simulation data
    template_seq : str, optional
        Template sequence to extract rates for (default: "AAAAAAA")
    mutant_seq : str, optional
        Mutant sequence to extract rates for (default: "BBBBBBB")

    Returns
    -------
    dict
        Dictionary containing:
        - 'x': np.ndarray - One-hot encoded sequences, shape (n_samples, seq_length, 2)
        - 'y': np.ndarray - Raw activity values, shape (n_samples,)
        - 'ka_mat': np.ndarray - King-Altman pattern matrix
        - 'ka_contrib': np.ndarray - Rate contribution matrix
        - 'target_rates': list - Rates for template sequence
        - 'mut_rates': list - Rates for mutant sequence
        - 'sequences': np.ndarray - Original sequences as strings

    Examples
    --------
    >>> data = load_kinetic_data("ring_datasets/test_4ring_random_0.h5")
    >>> x_train, y_train = data['x'], data['y']
    >>> print(f"Loaded {len(x_train)} sequences of length {x_train.shape[1]}")
    """
    with h5py.File(filepath, "r") as f:
        # Load sequence and activity data
        sequences = f["data"]["seq"][:].astype(str)
        y = f["data"]["raw_activity"][:]

        # Load King-Altman matrices
        ka_mat = f["ka_pattern_matrix"][:]
        ka_contrib = f["rate_contrib_matrix"][:]

        # Find target and mutant rows

        target_row = np.where(sequences == template_seq)[0][0] if template_seq else 0
        mut_row = np.where(sequences == mutant_seq)[0][0] if mutant_seq else 1

        # Extract rates (columns 1 through -2, excluding seq and activities)
        target_rates = list(f["data"][target_row])[1:-2]
        mut_rates = list(f["data"][mut_row])[1:-2]
        dtypes = f["data"].dtype

    # One-hot encode sequences
    enc = OneHotEncoder(categories=[["A", "B"]], sparse=False)

    n_samples = len(sequences)
    seq_length = len(sequences[0])

    # Flatten sequences for encoding
    flat = np.array([list(str(seq)) for seq in sequences]).flatten()[:, None]
    encoded = enc.fit_transform(flat)

    # Reshape to (n_samples, seq_length, 2)
    x = encoded.reshape(n_samples, seq_length, 2)

    return {
        "x": x,
        "y": y,
        "ka_mat": ka_mat,
        "ka_contrib": ka_contrib,
        "target_rates": target_rates,
        "mut_rates": mut_rates,
        "dtype": dtypes,
    }
