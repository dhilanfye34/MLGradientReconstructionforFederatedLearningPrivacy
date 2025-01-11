import numpy as np

# Fake function to simulate gradient computation
def compute_gradients_numpy():
    # Example of simulated gradients for a simple model
    # Replace this with your actual logic for generating gradients
    gradients = [
        np.random.randn(64, 3, 7, 7),  # Example gradient for a convolutional layer
        np.random.randn(64),          # Example gradient for a bias term
        np.random.randn(128, 64, 3, 3),
        np.random.randn(128),
    ]
    return gradients

if __name__ == "__main__":
    # Compute gradients
    gradients = compute_gradients_numpy()

    # Save gradients as a .npy file
    np.save("gradients.npy", gradients)
    print("Gradients saved to gradients.npy")
