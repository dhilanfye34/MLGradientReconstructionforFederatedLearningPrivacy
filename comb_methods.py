import torch
import torch.nn.functional as F
from torch.autograd import grad
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image        
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


def combined_gradient_matching(model, origin_grad, switch_iteration=20, use_tv=True):
    """
    Combined gradient matching: switches from DLG to cosine-based reconstruction.
    """
    print("Debug: Entered combined_gradient_matching function...")
    print(f"Switch iteration: {switch_iteration}, Use TV: {use_tv}")

    # Debug: Verify gradient shapes and devices
    print("Debug: Checking origin_grad details...")
    for i, grad in enumerate(origin_grad):
        print(f"Origin gradient {i}: Shape = {grad.shape}, Device = {grad.device}")

    # Initialize dummy data and dummy labels
    dummy_data = torch.randn(origin_grad[0].size(), requires_grad=True, device=origin_grad[0].device)
    dummy_label = torch.tensor([243], device=origin_grad[0].device)  # Fixed label for German Shepherd

    print("Debug: Initialized dummy_data and dummy_label.")
    print(f"Dummy data shape: {dummy_data.shape}, Device: {dummy_data.device}")
    print(f"Dummy label: {dummy_label}")

    # Set up optimizer and reconstructor
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
    reconstructor = GradientReconstructor(model, mean_std=(0.0, 1.0), config={'cost_fn': 'sim'}, num_images=1)

    # Start the optimization loop
    print("Starting Combined Gradient Matching...")
    for iteration in range(20):  # Limit to 20 iterations for quick testing
        print(f"\n--- Iteration {iteration} ---")  # Debug: iteration counter

        def closure():
            print(f"Iteration {iteration}: Inside closure function.")
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label)
            print(f"Iteration {iteration}: Dummy loss = {dummy_loss.item()}")

            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)
            print(f"Iteration {iteration}: Computed dummy gradients.")

            # Debug: Show the first few dummy gradient norms
            for i, dg in enumerate(dummy_grad[:3]):
                print(f"Dummy gradient {i} norm: {dg.norm().item()}")

            # Use DLG for the first iterations
            if iteration < switch_iteration:
                print(f"Iteration {iteration}: Using dummy DLG gradient difference...")
                grad_diff = torch.tensor(0.0, device=origin_grad[0].device, requires_grad=True)
            else:
                print(f"Iteration {iteration}: Using dummy GradientReconstructor gradient difference...")
                grad_diff = torch.randn(1, device=origin_grad[0].device).abs().sum()

            # Apply TV regularization if enabled
            if use_tv:
                tv_loss = TV(dummy_data) * 1e-1
                grad_diff += tv_loss
                print(f"Iteration {iteration}: TV Regularization = {tv_loss.item()}")

            print(f"Iteration {iteration}: Gradient Difference = {grad_diff.item()}")  # Debug gradient difference
            grad_diff.backward()
            return grad_diff

        print(f"Iteration {iteration}: Before optimizer step...")
        optimizer.step(closure)
        print(f"Iteration {iteration}: After optimizer step.")

        # Print intermediate progress and save dummy data every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Saving reconstructed image...")
            save_image(dummy_data.clone().detach(), f"reconstructed_iter_{iteration}.png")

    print("Gradient Matching Complete!")
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
    target = torch.tensor([243])  # Target class: 243 (German Shepherd in ImageNet)
    loss = F.cross_entropy(model(input_image), target)
    origin_grad = grad(loss, model.parameters(), create_graph=True)

    # Perform combined gradient matching
    iteration = 0  # Starting iteration
    dummy_data, dummy_label = combined_gradient_matching(model, origin_grad, iteration)

    # Save and visualize results
    output_image_path = "11794_Combined_output.png"
    torch.save(dummy_data, output_image_path)
    print(f"Reconstructed image saved to {output_image_path}")
