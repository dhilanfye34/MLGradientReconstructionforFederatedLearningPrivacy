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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    image = Image.open(file_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension



def combined_gradient_matching(model, origin_grad, iteration, switch_iteration=1000, use_tv=True):
    """
    Combined method that switches between DLG (magnitude-based) and GradientReconstructor (direction-based) approaches.

    Arguments:
        model: The neural network model we are reconstructing input data for.
        origin_grad: The original gradients we are trying to match (must be a list of tensors).
        iteration: Current iteration in the optimization loop (used to decide switching point).
        switch_iteration: Iteration number at which to switch from DLG to GradientReconstructor.
        use_tv: Boolean indicating whether to apply Total Variation regularization to smooth the dummy image.

    Returns:
        dummy_data: The reconstructed input data that matches the origin gradients.
        dummy_label: The reconstructed label that matches the origin gradients.
    """
    # Validate origin_grad
    if not isinstance(origin_grad, list) or not all(isinstance(t, torch.Tensor) for t in origin_grad):
        raise ValueError("origin_grad must be a list of tensors.")

    # Initialize dummy data and labels
    dummy_data = torch.randn(origin_grad[0].size(), requires_grad=True)

    # Determine the output size for labels
    output_size = origin_grad[-1].shape[0] if len(origin_grad[-1].shape) > 0 else 1

    # Initialize dummy labels as integer class indices
    dummy_label = torch.randint(0, 1000, (1,), requires_grad=False)  # ImageNet has 1000 classes


    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.01)  # LBFGS optimizer
    reconstructor = GradientReconstructor(model, mean_std=(0.0, 1.0), config={'cost_fn': 'sim'}, num_images=1)

    for iteration in range(2000): # past 200-300 iterations, maybe not changing much, # wait parameter to see if you dont need to change to second approach, then experience with lambda, make sure 0 and 1 work and then also then try 0.5 and others
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label.argmax(dim=1))
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            if iteration < switch_iteration:
                grad_diff = deep_leakage_from_gradients(model, origin_grad)
            else:
                grad_diff = reconstructor._gradient_closure(optimizer, dummy_data, origin_grad, dummy_label)()

            if use_tv:
                grad_diff += TV(dummy_data) * 1e-1

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

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