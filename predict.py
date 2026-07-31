"""
Run inference on a single currency note image using the trained model.

Usage:
    python predict.py --image path/to/note.jpg
    python predict.py --image path/to/note.jpg --model saved_models/best_model.pth
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn_model import SimpleCNN


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = SimpleCNN(num_classes=len(class_names), img_size=img_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size


def preprocess_image(image_path: str, img_size: int):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor


def predict(image_path: str, model_path: str = "saved_models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, img_size = load_model(model_path, device)
    image_tensor = preprocess_image(image_path, img_size).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    print("All class probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {probabilities[i].item() * 100:.2f}%")

    return predicted_class, confidence.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real/fake for a currency note image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="saved_models/best_model.pth", help="Path to trained model")
    args = parser.parse_args()

    predict(args.image, args.model)
