import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

# Import the combined gradient matching function
from comb_methods import combined_gradient_matching
from inversefed.nn.models import construct_model  # For creating the ResNet18 model

# Step 1: Load the input image
def load_input_image(image_path):
    """Load and preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    input_image = transform(img).unsqueeze(0).requires_grad_(True)  # Add batch dimension and enable gradients
    return input_image

# Step 2: Load the model
def load_model():
    """Load the ResNet18 model."""
    model, _ = construct_model("ResNet18", num_classes=1000)
    model.eval()  # Set the model to evaluation mode
    return model

# Step 3: Compute the original gradients
def compute_gradients(model, input_image, target_label):
    """Compute gradients for the given input image and target label."""
    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    outputs = model(input_image)  # Forward pass
    loss = criterion(outputs, torch.tensor([target_label]))  # Calculate loss
    origin_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)  # Compute gradients
    return origin_grad

# Step 4: Test the combined gradient matching method
def test_combined_method():
    # Load the input image
    image_path = "11794_ResNet18_ImageNet_input.png"  # Update with the correct path to your input image
    input_image = load_input_image(image_path)

    # Load the model
    model = load_model()

    # Define the target label
    target_label = 243  # Example label (e.g., 243 = "German Shepherd" in ImageNet)

    # Compute the original gradients
    origin_grad = compute_gradients(model, input_image, target_label)

    # Run the combined gradient matching function
    dummy_data, dummy_label = combined_gradient_matching(
        model=model,
        origin_grad=origin_grad,
        iteration=0,
        switch_iteration=100,
        use_tv=True
    )

    # Save the reconstructed image
    save_image(dummy_data, "reconstructed_image.png")
    print("Reconstructed image saved as 'reconstructed_image.png'.")

# Run the test
if __name__ == "__main__":
    test_combined_method()
