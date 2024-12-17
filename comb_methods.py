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


def combined_gradient_matching(model, origin_grad, switch_iteration=1000, use_tv=True):
    """
    Combined gradient matching: switches from DLG to cosine-based reconstruction.
    """
    # Initialize dummy data and dummy labels
    dummy_data = torch.randn(origin_grad[0].size(), requires_grad=True, device=origin_grad[0].device)
    output_size = 1000  # ImageNet classes
    dummy_label = torch.randint(0, output_size, (1,), requires_grad=False, device=origin_grad[0].device)

    # Set up optimizer and reconstructor
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
    reconstructor = GradientReconstructor(model, mean_std=(0.0, 1.0), config={'cost_fn': 'sim'}, num_images=1)

    # Start the optimization loop
    print("Starting Combined Gradient Matching...")
    for iteration in range(200):  # Start with 200 iterations
        print(f"--- Iteration {iteration} ---")  # Debug: iteration counter
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label)
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            # Switch between DLG and GradientReconstructor
            if iteration < switch_iteration:
                print(f"Iteration {iteration}: Using DLG method...")
                grad_diff = deep_leakage_from_gradients(model, origin_grad)
            else:
                print(f"Iteration {iteration}: Using GradientReconstructor method...")
                grad_diff = reconstructor._gradient_closure(optimizer, dummy_data, origin_grad, dummy_label)()

            # Apply TV regularization if enabled
            if use_tv:
                tv_loss = TV(dummy_data) * 1e-1
                grad_diff += tv_loss
                print(f"Iteration {iteration}: TV Regularization = {tv_loss.item()}")

            grad_diff.backward()
            print(f"Iteration {iteration}: Gradient Difference = {grad_diff.item()}")  # Debug gradient difference
            return grad_diff

        # Perform optimization step
        optimizer.step(closure)

        # Print intermediate progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Reconstruction progress...")
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
