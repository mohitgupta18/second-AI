import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from train_cnn import SmallCNN

def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--image", type=str, required=True, help="Path to image file")
    Parser.add_argument("--out-dir", type=str, default="outputs", help="Where best_model.pth & class_names.txt live")
    args = Parser.parse_args()

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    model_path = out_dir / "best_model.pth"

    with open(out_dir / "class_names.txt") as f:
        class_names = [x.strip() for x in f.readlines()]

    ckpt = torch.load(model_path, map_location=Device)
    in_channels = ckpt.get("in_channels", 3)
    num_classes = len(class_names)

    model = SmallCNN(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(Device).eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406][:in_channels],
                             std=[0.229,0.224,0.225][:in_channels])
    ])

    img = Image.open(args.image).convert("RGB")
    Tensor = transform(img).unsqueeze(0).to(Device)
    with torch.no_grad():
        logits = model(Tensor)
        pred = torch.argmax(logits, dim=1).item()

    print(f"✅ Predicted Class: {class_names[pred]}")

if __name__ == "__main__":
    Main()
