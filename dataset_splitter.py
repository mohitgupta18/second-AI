import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC_DIR = Path("data/custom")      # source: class subfolders directly inside
DST_DIR = Path("data/split_custom")# destination: will create train/val/test

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

def MakeDir(p):
    p.mkdir(parents=True, exist_ok=True)

for cls in os.listdir(SRC_DIR):
    cls_path = SRC_DIR / cls
    if not cls_path.is_dir():
        continue

    Images = [p for p in cls_path.iterdir() if p.is_file()]
    trainimgs, tempimgs = train_test_split(Images, test_size=(1-train_ratio), random_state=42, shuffle=True)
    val_imgs, test_imgs = train_test_split(tempimgs, test_size=test_ratio/(test_ratio+val_ratio), random_state=42, shuffle=True)

    for split, split_imgs in zip(["train", "val", "test"], [trainimgs, val_imgs, test_imgs]):
        dst = DST_DIR / split / cls
        MakeDir(dst)
        for img in split_imgs:
            shutil.copy(img, dst / img.name)

print(f"✅ Dataset split into 70/15/15 at: {DST_DIR.resolve()}")
