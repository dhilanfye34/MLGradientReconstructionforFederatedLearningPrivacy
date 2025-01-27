import numpy as np

def serialize_gradients(gradients):
    """
    Converts gradients (NumPy array) into bytes for transmission.
    """
    return gradients.tobytes()

def deserialize_gradients(data):
    """
    Converts bytes back into NumPy array for processing.
    """
    return np.frombuffer(data, dtype=np.float32)

def process_gradients(gradients):
    """
    Example function to process gradients on the server.
    This is just a placeholder for any real gradient processing logic.
    """
    print("Processing gradients...")
    # Example: Add 1 to each gradient value
    return gradients + 1
