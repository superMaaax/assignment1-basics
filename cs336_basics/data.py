from __future__ import annotations

import numpy as np
import torch
import numpy.typing as npt


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Calculate valid starting positions
    # We need context_length + 1 tokens total (input + target), so max start is len - context_length - 1
    max_start_idx = len(dataset) - context_length
    
    # Sample random starting indices
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input and target sequences
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)
    
    for i, start_idx in enumerate(start_indices):
        # Input sequence: tokens at positions [start_idx, start_idx + context_length)
        x[i] = dataset[start_idx:start_idx + context_length]
        # Target sequence: tokens at positions [start_idx + 1, start_idx + context_length + 1)
        # This is the "next token" for each position in the input
        y[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and move to device
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    
    return x_tensor, y_tensor