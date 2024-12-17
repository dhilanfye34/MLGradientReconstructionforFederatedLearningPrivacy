import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights
from comb_methods import combined_gradient_matching
from inversefed import construct_dataloaders, utils, consts
from PIL import Image
from torchvision import transforms

# Step 1: Define TrainingStrategy class
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

# Load normalization constants for ImageNet
dm = torch.as_tensor(consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(consts.imagenet_std, **setup)[:, None, None]

# Step 3: Helper Functions
def plot(tensor, title, save_path=None):
    """Helper function to plot and save images."""
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    tensor_to_plot = tensor[0].permute(1, 2, 0).cpu()
    plt.imshow(tensor_to_plot)
    plt.title(title)
    if save_path:
        save_image(tensor, save_path)
    plt.show()

# Step 4: Test Combined Gradient Matching
def test_combined_method():
    # Load pretrained ResNet18
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.to(**setup)
    model.eval()

    # Load the input image
    input_image_path = "images/11794_ResNet18_ImageNet_input.png"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(input_image_path).convert("RGB")
    ground_truth = transform(image).unsqueeze(0).to(**setup)

    # Target label
    label = torch.tensor([243], device=setup['device'])  # German Shepherd class
    plot(ground_truth, f"Ground Truth (Label: {label})", "11794_input_image.png")

    # Compute gradients
    model.zero_grad()
    target_loss = torch.nn.functional.cross_entropy(model(ground_truth), label)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]

    # Run combined gradient matching
    print("Starting Combined Gradient Matching...")
    dummy_data, dummy_label = combined_gradient_matching(
        model=model,
        origin_grad=input_gradient, 
        use_tv=True
    )

    # Save and visualize reconstructed image
    plot(dummy_data, "Reconstructed (Combined)", "11794_Combined_output.png")
    print("Reconstructed image saved successfully.")

# Run the test
if __name__ == "__main__":
    test_combined_method()
