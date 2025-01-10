import numpy as np

def compute_gradients():
    # Simulate gradient computation with NumPy
    inputs = np.random.randn(10, 3 * 32 * 32)  # Mock input data
    weights = np.random.randn(3 * 32 * 32, 10)  # Mock model weights
    labels = np.random.randint(0, 10, size=(10,))  # Mock labels

    # Compute "gradients"
    logits = np.dot(inputs, weights)  # Forward pass
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
    gradients = np.dot(inputs.T, probabilities - np.eye(10)[labels])  # Simplified gradients

    # Save gradients to file
    np.save("gradients.npy", gradients)
    print("Gradients saved to gradients.npy")

if __name__ == "__main__":
    compute_gradients()
