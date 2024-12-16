import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights
from comb_methods import combined_gradient_matching
from inversefed import construct_dataloaders, utils, consts
from inversefed.reconstruction_algorithms import GradientReconstructor
from dlg_original import deep_leakage_from_gradients
from PIL import Image
from torchvision import transforms  


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
    # Load the input image
    input_image_path = "11794_ResNet18_ImageNet_input.png"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    image = Image.open(input_image_path).convert("RGB")
    ground_truth = transform(image).unsqueeze(0).to(**setup)

    # Define the target label (German Shepherd class in ImageNet = 243)
    label = torch.tensor([243], device=setup['device'])

    # Visualize and save the ground truth image
    plot(ground_truth, f"Ground Truth (Label: {label})", "11794_input_image.png")

    # Compute the gradients
    model.zero_grad()
    target_loss = torch.nn.functional.cross_entropy(model(ground_truth), label)
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
        switch_iteration=1000,
        use_tv=True
    )

    # Save and visualize reconstructed images
    plot(dummy_data, "Reconstructed (Combined)", "11794_Combined_output.png")

# Run the test
if __name__ == "__main__":
    test_combined_method()
