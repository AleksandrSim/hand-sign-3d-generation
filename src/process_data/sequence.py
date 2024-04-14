import numpy as np


def filter_non_zero(seq: np.ndarray) -> np.ndarray:
    """Filters the input array to remove trailing zeros across all dimensions.

    The process is based on the first occurrence of an all-zero slice in the
    last dimension.

    Args:
        seq (np.ndarray): A numpy array with an expected shape of (3, N, T),
            where N is the number of points and T is the number of time steps.

    Returns:
        np.ndarray: The filtered numpy array with trailing zeros removed in the
            last dimension.

    """
    non_zero_mask = np.all(seq == 0, axis=(0, 1))
    index = np.argmax(non_zero_mask)
    seq = seq[:, :, :index]
    return seq
