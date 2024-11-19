import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights
from comb_methods import combined_gradient_matching
from inversefed import construct_dataloaders, utils, consts
from inversefed.reconstruction_algorithms import GradientReconstructor
from dlg_original import deep_leakage_from_gradients


# Step 1: Define TrainingStrategy class for defs
class TrainingStrategy:
    def __init__(self, augmentations=None):
        self.augmentations = augmentations
        self.optimizer = "adam"
        self.lr = 0.1
        self.batch_size = 128
        self.epochs = 90
        self.scheduler = "cos"


# Step 2: System Setup
setup = utils.system_startup()
defs = TrainingStrategy(augmentations=None)

# Load the dataset
loss_fn, trainloader, validloader = construct_dataloaders('CIFAR10', defs)

# Load normalization constants for ImageNet
dm = torch.as_tensor(consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(consts.imagenet_std, **setup)[:, None, None]

# Step 3: Helper Functions
def plot(tensor, title, save_path=None):
    """Helper function to plot and save images."""
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    plt.title(title)
    if save_path:
        save_image(tensor, save_path)
        print(f"Saved image to {save_path}")
    plt.show()


# Step 4: Test Combined Gradient Matching
def test_combined_method():
    # Load pretrained ResNet18
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(**setup)
    model.eval()

    # Select an image from the dataset
    idx = 1  # Change this index based on your dataset (e.g., CIFAR-10 or ImageNet)
    img, label = validloader.dataset[idx]
    labels = torch.as_tensor((label,), device=setup['device'])
    ground_truth = img.to(**setup).unsqueeze(0)

    # Visualize and save the ground truth image
    plot(ground_truth, f"Ground Truth (Label: {label})", f"{idx}_input_image.png")

    # Compute the gradients
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]

    # Debug: Print gradient shapes
    print("Input Gradient Shapes:")
    for i, grad in enumerate(input_gradient):
        print(f"Gradient {i}: {grad.shape}")

    # Step 5: Run the Combined Method
    print("Testing Combined Gradient Matching...")
    dummy_data, dummy_label = combined_gradient_matching(
        model=model,
        origin_grad=input_gradient,
        iteration=0,
        switch_iteration=300,
        use_tv=True
    )

    # Save and visualize reconstructed images
    plot(dummy_data, "Reconstructed (Combined)", f"{idx}_Combined_output.png")

# Run the test
if __name__ == "__main__":
    test_combined_method()
