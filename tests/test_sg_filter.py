import torch


def savitzky_golay_coeffs(window_size, poly_order):
    """
    Compute the Savitzky-Golay filter coefficients using PyTorch.

    Parameters:
    - window_size: The size of the window (must be odd).
    - poly_order: The order of the polynomial to fit.

    Returns:
    - coeffs: The filter coefficients as a PyTorch tensor.
    """
    # Ensure the window size is odd and greater than the polynomial order
    if window_size % 2 == 0 or window_size <= poly_order:
        raise ValueError("window_size must be odd and greater than poly_order.")

    # Generate the Vandermonde matrix of powers for the polynomial fit
    half_window = (window_size - 1) // 2
    indices = torch.arange(-half_window, half_window + 1, dtype=torch.float32)
    A = torch.vander(indices, N=poly_order + 1, increasing=True)

    # Compute the pseudo-inverse of the matrix
    pseudo_inverse = torch.linalg.pinv(A)

    # The filter coefficients are given by the first row of the pseudo-inverse
    coeffs = pseudo_inverse[0]

    return coeffs


# Generate sample data: a noisy sine wave
torch.manual_seed(0)  # For reproducibility
x = torch.linspace(0, 10, 100)
y = torch.sin(x) + torch.randn(100) * 0.5


# Apply the Savitzky-Golay filter
def apply_savitzky_golay(y, coeffs):
    """
    Apply the Savitzky-Golay filter to a 1D signal using the provided coefficients.

    Parameters:
    - y: The input signal as a PyTorch tensor.
    - coeffs: The filter coefficients as a PyTorch tensor.

    Returns:
    - y_filtered: The filtered signal.
    """
    # Pad the signal at both ends to handle the borders
    pad_size = len(coeffs) // 2
    y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])

    # Apply convolution
    y_filtered = torch.conv1d(
        y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid"
    )

    return y_filtered.view(-1)


window_size = 15
poly_order = 5
coeffs = savitzky_golay_coeffs(window_size, poly_order)
y_filtered = apply_savitzky_golay(y, coeffs)

# Plot the original and filtered signals
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), label="Original", alpha=0.5)
plt.plot(x.numpy(), y_filtered.numpy(), label="Filtered", linewidth=2)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Savitzky-Golay Filter Application")
plt.show()
