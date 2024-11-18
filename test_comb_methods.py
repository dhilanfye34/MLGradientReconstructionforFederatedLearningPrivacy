import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from comb_methods import combined_gradient_matching
from inversefed import construct_dataloaders, utils, consts
from inversefed.reconstruction_algorithms import GradientReconstructor
from dlg_original import deep_leakage_from_gradients


# Step 1: System Setup
setup = utils.system_startup()

# Manually define the strategy since `training_strategy` is missing
defs = {
    "optimizer": "adam",
    "lr": 0.1,
    "batch_size": 128,
    "epochs": 90,
    "scheduler": "cos",
}

# Load the dataset
loss_fn, trainloader, validloader = construct_dataloaders('ImageNet', defs, data_path='/data/imagenet')

# Load normalization constants for ImageNet
dm = torch.as_tensor(consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(consts.imagenet_std, **setup)[:, None, None]

# Step 2: Helper Functions
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


# Step 3: Test Combined Gradient Matching
def test_combined_method():
    # Load pretrained ResNet18
    model = torchvision.models.resnet18(pretrained=True)
    model.to(**setup)
    model.eval()

    # Select the beagle image from the dataset
    idx = 8112  # Beagle's index in ImageNet
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

    # Step 4: Test Different Reconstruction Methods
    methods = {
        "DLG": deep_leakage_from_gradients,
        "GradientReconstructor": GradientReconstructor,
        "Combined": combined_gradient_matching
    }
    results = {}

    for method_name, method in methods.items():
        print(f"Testing {method_name}...")
        
        if method_name == "DLG":
            reconstructed_image, _ = method(model, input_gradient, labels)
        
        elif method_name == "GradientReconstructor":
            config = dict(signed=True, boxed=True, cost_fn='sim', lr=0.1, optim='adam', max_iterations=24000)
            reconstructor = method(model, (dm, ds), config, num_images=1)
            reconstructed_image, stats = reconstructor.reconstruct(input_gradient, labels, img_shape=(3, 224, 224))
        
        elif method_name == "Combined":
            reconstructed_image, _ = method(
                model=model,
                origin_grad=input_gradient,
                iteration=0,
                switch_iteration=100,
                use_tv=True
            )
        
        # Save and visualize reconstructed images
        plot(reconstructed_image, f"Reconstructed ({method_name})", f"{idx}_{method_name}_output.png")
        results[method_name] = reconstructed_image

    # Step 5: Compare Results
    print("\nComparison of Reconstructed Images:")
    for method_name, image in results.items():
        mse = (image.detach() - ground_truth).pow(2).mean()
        print(f"{method_name} MSE: {mse:.4f}")


# Run the test
if __name__ == "__main__":
    test_combined_method()
