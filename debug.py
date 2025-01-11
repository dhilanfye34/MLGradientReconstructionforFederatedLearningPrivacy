import numpy as np

gradients = np.load("gradients.npy", allow_pickle=True)
print([g.shape for g in gradients])  # Verify shapes of saved gradients
