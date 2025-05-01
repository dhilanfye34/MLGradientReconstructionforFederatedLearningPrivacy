import os
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import save_image
from inversefed.reconstruction_algorithms import GradientReconstructor
from inversefed.metrics import total_variation as TV
from cnn_model import SmallCNN  # Use local CIFAR-10 model


def combined_gradient_matching(model, origin_grad, label, switch_iteration=10, use_tv=True, debug=False):
    """
    Combined gradient matching: switches from DLG (L2) to cosine-based reconstruction.
    Prints minimal output unless debug=True.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize dummy data and labels
    dummy_data = torch.randn((1, 3, 32, 32), requires_grad=True, device=origin_grad[0].device)
    dummy_label = torch.tensor([label.item()], device=origin_grad[0].device)

    optimizer = torch.optim.LBFGS([dummy_data], lr=0.01)

    for iteration in range(20):
        if iteration % 5 == 0:
            print(f"🔁 Iteration {iteration}...")

        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, dummy_label)

            dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            if iteration < switch_iteration:
                grad_diff = sum((dg - og).norm() for dg, og in zip(dummy_gradients, origin_grad) if dg.shape == og.shape)
            else:
                grad_diff = sum(1 - F.cosine_similarity(dg.flatten(), og.flatten(), dim=0)
                                for dg, og in zip(dummy_gradients, origin_grad) if dg.shape == og.shape)

            if use_tv:
                tv_loss = TV(dummy_data) * 1e-2
                grad_diff = grad_diff + tv_loss

            if debug and iteration % 10 == 0:
                print(f"   Grad diff: {grad_diff.item():.4f}")

            grad_diff.requires_grad_()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iteration % 10 == 0:
            mean = torch.tensor([0.5, 0.5, 0.5], device=dummy_data.device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=dummy_data.device).view(1, 3, 1, 1)
            normalized_data = (dummy_data * std + mean).clamp(0, 1)
            save_image(normalized_data.clone().detach(), f"results/reconstructed_iter_{iteration}.png")
            print(f"💾 Saved image for iteration {iteration}")

    print("✅ Gradient Matching Complete!")
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
