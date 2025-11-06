import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, drop_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(drop_p),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(drop_p),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(drop_p),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def save_class_names(class_names, out_dir):
    with open(os.path.join(out_dir, "class_names.txt"), "w") as f:
        for c in class_names:
            f.write(str(c) + "\n")

def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(out_dir, "history_loss.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(out_dir, "history_acc.png"), bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc"); ax2.legend(); ax2.set_title("Accuracy")
    fig.suptitle("Training vs Validation")
    fig.savefig(os.path.join(out_dir, "history.png"), bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, out_dir):
    fig = plt.figure(figsize=(8,6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

def save_prediction_grid(images, labels, preds, class_names, out_dir, max_images=25):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)

    imgs = images[:max_images].cpu().numpy()
    imgs = (imgs * std + mean).clip(0,1)

    labs = labels[:max_images].cpu().numpy()
    prds = preds[:max_images].cpu().numpy()

    n = min(max_images, len(imgs))
    cols = 5
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols*2.5, rows*2.5))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i+1)
        img = np.transpose(imgs[i], (1,2,0))
        ax.imshow(img)
        title = f"P:{class_names[prds[i]]}\\nT:{class_names[labs[i]]}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample Predictions")
    fig.savefig(os.path.join(out_dir, "predictions_grid.png"), bbox_inches="tight")
    plt.close(fig)

def get_transforms(img_size=32, in_channels=3):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406][:in_channels],
                                     std=[0.229,0.224,0.225][:in_channels])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4) if img_size >= 32 else transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    return train_tf, test_tf

def load_dataset(name, data_root, img_size, val_split=0.15):
    name = name.lower()
    if name == "cifar10":
        train_tf, test_tf = get_transforms(img_size=32, in_channels=3)
        trainset_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        class_names = trainset_full.classes

        total = len(trainset_full)
        val_len = int(total * val_split)
        train_len = total - val_len
        trainset, valset = random_split(trainset_full, [train_len, val_len],
                                        generator=torch.Generator().manual_seed(42))
        valset.dataset.transform = test_tf

        in_channels = 3
        return trainset, valset, testset, class_names, in_channels, 32

    if name == "fashion_mnist":
        train_tf, test_tf = get_transforms(img_size=32, in_channels=1)
        trainset_full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
        testset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=test_tf)
        class_names = trainset_full.classes

        total = len(trainset_full)
        val_len = int(total * val_split)
        train_len = total - val_len
        trainset, valset = random_split(trainset_full, [train_len, val_len],
                                        generator=torch.Generator().manual_seed(42))
        valset.dataset.transform = test_tf

        in_channels = 1
        return trainset, valset, testset, class_names, in_channels, 32

    if name == "custom":
        train_tf, test_tf = get_transforms(img_size=img_size, in_channels=3)
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        test_dir = os.path.join(data_root, "test")

        trainset = datasets.ImageFolder(train_dir, transform=train_tf)
        valset = datasets.ImageFolder(val_dir, transform=test_tf)
        testset = datasets.ImageFolder(test_dir, transform=test_tf)
        class_names = trainset.classes
        in_channels = 3
        return trainset, valset, testset, class_names, in_channels, img_size

    raise ValueError("Unknown dataset. Use: cifar10 | fashion_mnist | custom")

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, leave=False, desc="Train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy_from_logits(outputs, targets) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, leave=False, desc="Val/Test"):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy_from_logits(outputs, targets) * images.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for images, targets in tqdm(loader, leave=False, desc="Eval"):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    import numpy as _np
    all_preds = _np.concatenate(all_preds)
    all_targets = _np.concatenate(all_targets)
    return all_preds, all_targets

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "fashion_mnist", "custom"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    trainset, valset, testset, class_names, in_channels, eff_img = load_dataset(
        args.dataset, args.data_root, args.img_size
    )
    save_class_names(class_names, out_dir)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    num_classes = len(class_names)
    model = SmallCNN(num_classes=num_classes, in_channels=in_channels, drop_p=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None
    early = EarlyStopping(patience=args.patience, min_delta=0.0)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_path = out_dir / "best_model.pth"

    print(f"Device: {device} | Classes: {num_classes} | Image: {eff_img}x{eff_img}")
    print("Training...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "in_channels": in_channels}, best_path)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"time={elapsed:.1f}s")

        if early.step(val_loss):
            print("Early stopping triggered.")
            break

    plot_history(history, out_dir)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Evaluating on test set...")
    preds, targets = evaluate_full(model, test_loader, device)
    acc = (preds == targets).mean()

    report = classification_report(targets, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_names, out_dir)

    test_batch = next(iter(test_loader))
    imgs, labs = test_batch
    with torch.no_grad():
        logits = model(imgs.to(device))
        pred_batch = torch.argmax(logits, dim=1).cpu()
    save_prediction_grid(imgs, labs, pred_batch, class_names, out_dir, max_images=25)

    summary = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "final_test_accuracy": float(acc),
        "precision_recall_f1_per_class": report,
        "best_val_loss": float(best_val),
        "epochs_ran": len(history["train_loss"])
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    example = torch.randn(1, in_channels, eff_img, eff_img).to(device)
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, out_dir / "model_scripted.pt")

    print("\\n=== Final Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    from sklearn.metrics import classification_report as _cr
    print(_cr(targets, preds, target_names=class_names, zero_division=0))
    print(f"\\nArtifacts saved in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
