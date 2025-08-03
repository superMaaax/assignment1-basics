from __future__ import annotations
from jaxtyping import Float, Int
from collections.abc import Iterable
from torch import Tensor
import torch


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = torch.max(in_features, dim=dim, keepdim=True)[0]
    x_shift = in_features - x_max
    exp_x = torch.exp(x_shift)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Compute log softmax for numerical stability
    # log_softmax(x) = x - log(sum(exp(x))) = x - max(x) - log(sum(exp(x - max(x))))
    x_max = torch.max(inputs, dim=1, keepdim=True)[0]
    x_shifted = inputs - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    log_sum_exp_x = torch.log(sum_exp_x) + x_max
    log_softmax = inputs - log_sum_exp_x

    # Gather the log probabilities for the target classes
    batch_size = inputs.shape[0]
    batch_indices = torch.arange(batch_size, device=inputs.device)
    log_probs = log_softmax[batch_indices, targets]

    # Cross entropy is the negative log likelihood
    cross_entropy_loss = -log_probs

    # Return the mean across the batch
    return torch.mean(cross_entropy_loss)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Get all parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # Compute total norm of all gradients
    total_norm = 0.0
    for p in params_with_grad:
        param_norm = p.grad.norm()
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    # Compute clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    # Only clip if necessary (total_norm > max_l2_norm)
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)
