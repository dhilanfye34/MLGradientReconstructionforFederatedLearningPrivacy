import torch
import torch.nn.functional as F
from torch.autograd import grad
from PIL import Image
from torchvision import transforms
from dlg_original import deep_leakage_from_gradients  # Importing original DLG method
from inversefed.reconstruction_algorithms import GradientReconstructor  # Importing InverseFed method
from inversefed.metrics import total_variation as TV  # Importing Total Variation (TV) regularization


def load_image(file_path):
    """Load and preprocess an image for use in the reconstruction pipeline."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet18 expects 224x224 images
        transforms.ToTensor()
    ])
    image = Image.open(file_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


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

    # Debug: Check if origin_grad is a list of tensors
    print("Debug: Checking origin_grad...")
    if not isinstance(origin_grad, list) or not all(isinstance(g, torch.Tensor) for g in origin_grad):
        raise ValueError("origin_grad must be a list of tensors.")
    print("Debug: origin_grad is valid.")

    # Initialize dummy data
    dummy_data = torch.randn(origin_grad[0].size(), requires_grad=True)

    # Debug: Ensure origin_grad[-1] is a tensor with valid dimensions
    if len(origin_grad[-1].size()) < 2:
        raise ValueError("origin_grad[-1] must have at least two dimensions for labels.")

    # Initialize dummy labels
    dummy_label = torch.randn((1, origin_grad[-1].size(1)), requires_grad=True)

    # Set up optimizer
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # Initialize GradientReconstructor
    reconstructor = GradientReconstructor(model, mean_std=(0.0, 1.0), config={'cost_fn': 'sim'}, num_images=1)

    # Begin optimization loop
    for i in range(300):
        def closure():
            optimizer.zero_grad()  # Reset gradients to zero before each iteration

            # Forward pass: Calculate model predictions on dummy data
            dummy_pred = model(dummy_data)

            # Cross-entropy loss between dummy predictions and dummy labels
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label.argmax(dim=1))

            # Backpropagation: Compute gradients of dummy_loss with respect to model parameters
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            # Switch between methods based on iteration
            if iteration < switch_iteration:
                # Use DLG's L2 norm-based gradient matching (magnitude-based approach)
                grad_diff = deep_leakage_from_gradients(model, origin_grad)
            else:
                # Use InverseFed's cosine similarity-based approach (direction-based approach)
                grad_diff = reconstructor._gradient_closure(optimizer, dummy_data, origin_grad, dummy_label)()

            # Add Total Variation (TV) regularization if enabled
            if use_tv:
                grad_diff += TV(dummy_data) * 1e-1  # Adjust weight as needed

            grad_diff.backward()  # Backpropagate gradient difference
            return grad_diff

        # Update dummy data and labels
        optimizer.step(closure)

    # Return final reconstructed data and labels
    return dummy_data, dummy_label


if __name__ == "__main__":
    # Load the model
    from inversefed.nn.models import construct_model

    model, _ = construct_model("ResNet18", num_classes=1000)  # ResNet18 expects ImageNet (1000 classes)
    model.eval()

    # Load the input image
    input_image_path = "11794_ResNet18_ImageNet_input.png"
    input_image = load_image(input_image_path)

    # Generate dummy gradients for testing
    target = torch.randint(0, 1000, (1,))  # Random label for the example
    loss = F.cross_entropy(model(input_image), target)
    origin_grad = grad(loss, model.parameters(), create_graph=True)

    # Perform combined gradient matching
    iteration = 0  # Starting iteration
    dummy_data, dummy_label = combined_gradient_matching(model, origin_grad, iteration)

    # Save and visualize results
    output_image_path = "reconstructed_image.png"
    torch.save(dummy_data, output_image_path)
    print(f"Reconstructed image saved to {output_image_path}")
