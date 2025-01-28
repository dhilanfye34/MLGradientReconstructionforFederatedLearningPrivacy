import numpy as np

# Fake function to simulate gradient computation using NumPy
def compute_gradients_numpy():
    """
    Simulate gradient computation for layers in a neural network using NumPy.
    Replace this with actual gradient computation logic if needed.
    """
    # Example: Gradients for a convolutional layer (weights and bias)
    conv1_weights = np.random.randn(64, 3, 7, 7)  # 64 filters, 3 input channels, 7x7 kernel
    conv1_bias = np.random.randn(64)              # One bias per filter

    # Example: Gradients for another convolutional layer
    conv2_weights = np.random.randn(128, 64, 3, 3)  # 128 filters, 64 input channels, 3x3 kernel
    conv2_bias = np.random.randn(128)               # One bias per filter

    # Collect gradients into a list
    gradients = [conv1_weights, conv1_bias, conv2_weights, conv2_bias]
    return gradients

if __name__ == "__main__":
    # Compute gradients
    gradients = compute_gradients_numpy()

    # Print the shapes of the computed gradients for debugging
    print("Gradient shapes:")
    for i, grad in enumerate(gradients):
        print(f"Layer {i}: {grad.shape}")

    # Save each gradient with a unique name using np.savez
    gradient_dict = {f"layer_{i}": grad for i, grad in enumerate(gradients)}
    np.savez("gradients.npz", **gradient_dict)
    print("Gradients saved to gradients.npz")
