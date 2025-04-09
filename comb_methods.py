import os
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import save_image        
from inversefed.reconstruction_algorithms import GradientReconstructor
from inversefed.metrics import total_variation as TV
from cnn_model import SmallCNN  # Use local CIFAR-10 model

def combined_gradient_matching(model, origin_grad, switch_iteration=50, use_tv=True):
    """
    Combined gradient matching: switches from DLG (L2) to cosine-based reconstruction.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize dummy data and labels
    dummy_data = torch.randn((1, 3, 32, 32), requires_grad=True, device=origin_grad[0].device)
    dummy_label = torch.tensor([0] * dummy_data.size(0), device=origin_grad[0].device)  # Dummy label, will be overwritten

    # Set up optimizer
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.01)

    # Optimization loop
    for iteration in range(100):
        print(f"\n--- Iteration {iteration} ---")  # Iteration marker

        def closure():
            print(f"Iteration {iteration}: Inside closure function.")
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label)

            dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            print(f"Iteration {iteration}: Computed dummy gradients.")

            if iteration < switch_iteration:
                print(f"Iteration {iteration}: Using L2 gradient difference (DLG)...")
                grad_diff = sum((dg - og).norm() for dg, og in zip(dummy_gradients, origin_grad) if dg.shape == og.shape)
            else:
                print(f"Iteration {iteration}: Using Cosine Similarity-based matching...")
                grad_diff = sum(1 - F.cosine_similarity(dg.flatten(), og.flatten(), dim=0) for dg, og in zip(dummy_gradients, origin_grad) if dg.shape == og.shape)

            if use_tv:
                tv_loss = TV(dummy_data) * 1e-2
                grad_diff = grad_diff + tv_loss
                print(f"Iteration {iteration}: TV Regularization = {tv_loss.item()}")

            grad_diff.requires_grad_()
            print(f"Iteration {iteration}: Gradient Difference = {grad_diff.item()}")
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Saving reconstructed image...")
            mean = torch.tensor([0.5, 0.5, 0.5], device=dummy_data.device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=dummy_data.device).view(1, 3, 1, 1)
            normalized_data = (dummy_data * std + mean).clamp(0, 1)
            save_image(normalized_data.clone().detach(), f"results/reconstructed_iter_{iteration}.png")

    print("✅ Gradient Matching Complete!")
    print(f"Final Dummy Data Stats: Mean = {dummy_data.mean().item()}, Std = {dummy_data.std().item()}")
    return dummy_data, dummy_label

if __name__ == "__main__":
    import pickle
    import sys
    # Load sample image and label from socket or pickle for testing
    if len(sys.argv) == 2 and sys.argv[1].endswith(".pkl"):
        with open(sys.argv[1], "rb") as f:
            image, label = pickle.load(f)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
    else:
        # Fallback CIFAR-10 image
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        image, label = cifar10[0]
        image = image.unsqueeze(0)
        label = torch.tensor([label])

    model = SmallCNN(pretrained=True)
    model.eval()

    image = image.detach().requires_grad_(True)
    output = model(image)
    loss = F.cross_entropy(output, label)
    origin_grad = grad(loss, model.parameters(), create_graph=True)

    dummy_data, dummy_label = combined_gradient_matching(model, origin_grad)

    output_path = "results/final_reconstructed_image.png"
    save_image(dummy_data, output_path)
    print(f"🖼️ Final reconstructed image saved to: {output_path}")
