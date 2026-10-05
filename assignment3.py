"""
AI Accelerator Design - Assignment 3
BCHW <-> B x (CHW) Tensor Flattening and Reconstruction

The required tensor-layout conversions are implemented from scratch.
No reshape(), view(), flatten(), ravel(), or equivalent shape-conversion
function is used in the core algorithms.
"""

import numpy as np


def flatten_bchw(tensor):
    """
    Convert BCHW -> B x (CHW) using explicit indexing.

    For an element tensor[b, c, h, w], its flattened position is:
        k = c * H * W + h * W + w

    Parameters
    ----------
    tensor : numpy.ndarray
        Tensor with shape (B, C, H, W).

    Returns
    -------
    numpy.ndarray
        Tensor with shape (B, C*H*W).
    """
    B, C, H, W = tensor.shape
    flat = np.empty((B, C * H * W), dtype=tensor.dtype)

    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k = c * H * W + h * W + w
                    flat[b, k] = tensor[b, c, h, w]

    return flat


def reconstruct_bchw(flat, B, C, H, W):
    """
    Convert B x (CHW) -> BCHW using explicit indexing.

    Given flattened index k:
        c = k // (H*W)
        r = k % (H*W)
        h = r // W
        w = r % W

    Parameters
    ----------
    flat : numpy.ndarray
        Tensor with shape (B, C*H*W).
    B, C, H, W : int
        Original BCHW dimensions.

    Returns
    -------
    numpy.ndarray
        Reconstructed tensor with shape (B, C, H, W).
    """
    reconstructed = np.empty((B, C, H, W), dtype=flat.dtype)

    for b in range(B):
        for k in range(C * H * W):
            c = k // (H * W)
            r = k % (H * W)
            h = r // W
            w = r % W
            reconstructed[b, c, h, w] = flat[b, k]

    return reconstructed


def calculate_errors(original, reconstructed):
    """Return maximum absolute error and mean absolute error."""
    difference = original - reconstructed
    abs_difference = np.abs(difference)

    emax = np.max(abs_difference)
    mae = np.mean(abs_difference)

    return emax, mae


def make_tensor(B, C, H, W):
    """
    Generate a deterministic synthetic BCHW tensor.

    Synthetic tensors are allowed by the assignment for larger channel
    counts and represent intermediate DNN feature maps.
    """
    total = B * C * H * W
    values = np.arange(total, dtype=np.float32)

    # Give the tensor non-trivial floating-point values while preserving
    # exact storage/reconstruction for this experiment.
    values = (values * 0.125) - 7.25

    tensor = np.empty((B, C, H, W), dtype=np.float32)

    index = 0
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    tensor[b, c, h, w] = values[index]
                    index += 1

    return tensor


def manual_example():
    """
    Demonstrate the assignment's requested small example:
        B=1, C=2, H=2, W=3
    """
    B, C, H, W = 1, 2, 2, 3
    tensor = make_tensor(B, C, H, W)
    flat = flatten_bchw(tensor)

    print("\nManual indexing example: B=1, C=2, H=2, W=3")
    print("Formula: k = c*H*W + h*W + w")
    print()

    examples = [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 2),
        (0, 1, 0, 0),
        (0, 1, 1, 2),
    ]

    for b, c, h, w in examples:
        k = c * H * W + h * W + w
        print(
            f"I[{b},{c},{h},{w}] -> F[{b},{k}]   "
            f"value = {tensor[b,c,h,w]:.3f}"
        )

    print("\nFlattened row:")
    print(flat[0])


def run_experiments():
    # B, C, H, W
    configurations = [
        (1, 1, 28, 28),    # MNIST-like
        (1, 3, 32, 32),    # CIFAR-like
        (1, 8, 16, 16),
        (1, 16, 16, 16),
        (1, 32, 16, 16),
        (1, 64, 16, 16),
        (1, 128, 8, 8),
        (1, 256, 8, 8),
        (1, 500, 8, 8),
    ]

    print("\nExperimental Results")
    print("-" * 75)
    print(f"{'B':>4} {'C':>5} {'H':>5} {'W':>5} "
          f"{'Emax':>15} {'MAE':>15}")
    print("-" * 75)

    results = []

    for B, C, H, W in configurations:
        original = make_tensor(B, C, H, W)

        flat = flatten_bchw(original)
        reconstructed = reconstruct_bchw(flat, B, C, H, W)

        emax, mae = calculate_errors(original, reconstructed)

        results.append((B, C, H, W, emax, mae))

        print(
            f"{B:4d} {C:5d} {H:5d} {W:5d} "
            f"{emax:15.6e} {mae:15.6e}"
        )

    return results


if __name__ == "__main__":
    manual_example()
    run_experiments()
