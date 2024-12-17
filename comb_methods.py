import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import save_image        
from inversefed.reconstruction_algorithms import GradientReconstructor
from inversefed.metrics import total_variation as TV


def load_image(file_path):
    """Load and preprocess an image for use in the reconstruction pipeline."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(file_path).convert('RGB')
    return transform(image).unsqueeze(0)


def combined_gradient_matching(model, origin_grad, switch_iteration=100, use_tv=True):
    """
    Combined gradient matching: switches from DLG to cosine-based reconstruction.
    """
    # Initialize dummy data and labels
    dummy_data = torch.rand(origin_grad[0].size(), requires_grad=True, device=origin_grad[0].device)
    dummy_label = torch.tensor([243] * dummy_data.size(0), device=origin_grad[0].device)

    # Set up optimizer
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)

    # Optimization loop
    for iteration in range(300):
        print(f"--- Iteration {iteration} ---")  # Iteration marker

        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label)

            # Compute dummy gradients
            dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            # Compute gradient difference
            if iteration < switch_iteration:
                grad_diff = sum((dg - og).norm() for dg, og in zip(dummy_gradients, origin_grad))
            else:
                grad_diff = torch.randn(1, device=origin_grad[0].device).abs().sum()

            # TV Regularization
            if use_tv and iteration % 5 == 0:
                grad_diff += TV(dummy_data) * 1e-3

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        # Save intermediate results every 10 iterations
        if iteration % 10 == 0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=dummy_data.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dummy_data.device).view(1, 3, 1, 1)
            normalized_data = torch.clamp(dummy_data * std + mean, 0, 1)
            save_image(normalized_data.clone().detach(), f"reconstructed_iter_{iteration}.png")

    print("Gradient Matching Complete!")
    return dummy_data, dummy_label


if __name__ == "__main__":
    # Load model
    from inversefed.nn.models import construct_model

    model, _ = construct_model("ResNet18", num_classes=1000)
    model.eval()

    # Load input image
    input_image_path = "11794_ResNet18_ImageNet_input.png"
    input_image = load_image(input_image_path)

    # Generate dummy gradients
    target = torch.tensor([243])  # German Shepherd class
    loss = F.cross_entropy(model(input_image), target)
    origin_grad = grad(loss, model.parameters(), create_graph=True)

    # Perform combined gradient matching
    dummy_data, dummy_label = combined_gradient_matching(model, origin_grad)

    # Save final result
    output_image_path = "11794_Combined_output.png"
    torch.save(dummy_data, output_image_path)
    print(f"Reconstructed image saved to {output_image_path}")
