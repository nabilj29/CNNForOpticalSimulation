import os
from PIL import Image

import torch
import torchvision.transforms as T

from onnClass import ONNClassifier


def load_model(device: torch.device) -> ONNClassifier:
    """Recreate the ONN classifier and load trained weights."""
    edge_kernel = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    model = ONNClassifier(kernel=edge_kernel, num_classes=12).to(device)
    checkpoint = torch.load("checkpoints/onn_classifier.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def preprocess_image(path: str) -> torch.Tensor:
    """Match the training transforms (grayscale → resize 512 → tensor)."""
    transform = T.Compose(
        [
            T.Grayscale(),
            T.Resize((512, 512)),
            T.ToTensor(),
        ]
    )
    img = Image.open(path).convert("L")
    return transform(img)


def main():
    img_path = "../DSAD/abdominal_wall/04/image01.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    img_tensor = preprocess_image(img_path).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)

    predicted_class = probs.argmax(dim=1).item()
    prob = probs[0, predicted_class].item()

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {prob:.4f}")
    print("All class probabilities:\n", probs.cpu().numpy())


if __name__ == "__main__":
    main()
