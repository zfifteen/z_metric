import numpy as np

def rolling_predict(buffer: np.ndarray) -> np.ndarray:
    """
    Predict the next embedding vector using a simple AR(1) rule:
    next ≈ last + (last - second_last)

    Parameters
    ----------
    buffer : np.ndarray
        Array of shape (window_size, dims) holding the recent Z-embeddings.

    Returns
    -------
    np.ndarray
        Predicted embedding of shape (dims,).
    """
    if buffer.shape[0] < 2:
        # Not enough history: return the last known point
        return buffer[-1]

    last = buffer[-1]
    prev = buffer[-2]
    # Linear extrapolation
    return last + (last - prev)
