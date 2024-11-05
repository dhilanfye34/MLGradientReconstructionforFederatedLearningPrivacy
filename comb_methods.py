import torch
import torch.nn.functional as F
from torch.autograd import grad
from dlg_original import deep_leakage_from_gradients  # Importing original DLG method
from inversefed.reconstruction_algorithms import GradientReconstructor  # Importing InverseFed method
from inversefed.metrics import total_variation as TV  # Importing Total Variation (TV) regularization

def combined_gradient_matching(model, origin_grad, iteration, switch_iteration=100, use_tv=True):
    """
    Combined method that switches between DLG (magnitude-based) and GradientReconstructor (direction-based) approaches.

    Arguments:
        model: The neural network model we are reconstructing input data for.
        origin_grad: The original gradients we are trying to match.
        iteration: Current iteration in the optimization loop (used to decide switching point).
        switch_iteration: Iteration number at which to switch from DLG to GradientReconstructor.
        use_tv: Boolean indicating whether to apply Total Variation regularization to smooth the dummy image.

    Returns:
        dummy_data: The reconstructed input data that matches the origin gradients.
        dummy_label: The reconstructed label that matches the origin gradients.
    """

    # Initialize dummy data and labels with random values to start the reconstruction process.
    # These will be adjusted to make their gradients match the origin_grad.
    dummy_data = torch.randn(origin_grad[0].size())  # Random data with same size as input gradient
    dummy_label = torch.randn((1, origin_grad[-1].size(1)))  # Random label with appropriate dimensions

    # Set up an optimizer to iteratively update dummy_data and dummy_label.
    # LBFGS (Limited-memory BFGS) is a quasi-Newton optimizer suited for this type of iterative optimization.
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # Initialize an instance of GradientReconstructor from the InverseFed library.
    # This will be used later in the process for direction-based matching (cosine similarity).
    reconstructor = GradientReconstructor(model, mean_std=(0.0, 1.0), config={'cost_fn': 'sim'}, num_images=1)

    # Begin optimization loop: perform multiple iterations to refine dummy_data and dummy_label.
    for i in range(300):
        # Define the closure function required by LBFGS optimizer.
        def closure():
            optimizer.zero_grad()  # Reset gradients to zero before each iteration

            # Forward pass: Calculate model predictions on dummy data.
            dummy_pred = model(dummy_data)

            # Calculate the cross-entropy loss between dummy predictions and dummy labels.
            # This loss will guide the adjustment of dummy_data to match the origin_grad.
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label.argmax(dim=1))

            # Backpropagation: Compute gradients of dummy_loss with respect to model parameters.
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            # Check if we're before or after the switch iteration
            if iteration < switch_iteration:
                # Before switch: Use DLG's L2 norm-based gradient matching (magnitude-based approach)
                # Calculate the sum of squared differences (L2 norm) between dummy_grad and origin_grad.
                grad_diff = sum(((dummy_g - origin_g) ** 2).sum() for dummy_g, origin_g in zip(dummy_grad, origin_grad))
            else:
                # After switch: Use InverseFed's cosine similarity-based approach (direction-based approach)
                # Calculate cosine similarity between dummy_grad and origin_grad to focus on gradient direction.
                grad_diff = reconstructor._gradient_closure(optimizer, dummy_data, origin_grad, dummy_label)()

            # Optional: Add Total Variation (TV) regularization to encourage smoothness in the dummy data.
            # TV regularization helps reduce noise and makes the reconstructed image look more natural.
            if use_tv:
                grad_diff += TV(dummy_data) * 1e-1  # Weight of TV regularization can be adjusted as needed
            
            # Perform backpropagation on grad_diff to adjust dummy_data and dummy_label.
            grad_diff.backward()
            return grad_diff

        # Update dummy_data and dummy_label using LBFGS optimizer
        optimizer.step(closure)

    # Return the final reconstructed dummy data and dummy label after all iterations
    return dummy_data, dummy_label
