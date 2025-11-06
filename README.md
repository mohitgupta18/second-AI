# CNN Image Classification Project (PyTorch)

## Objective
Design, train, and evaluate a Convolutional Neural Network capable of classifying images into multiple categories. 
This repo demonstrates data preprocessing, model architecture, training optimization, and deployment readiness.

##  Features
- End-to-end data pipeline (train/val/test split)
- CNN with 3+ conv blocks, BatchNorm, Dropout, ReLU
- Adam optimizer, CrossEntropy loss
- Early stopping & LR scheduling (ReduceLROnPlateau)
- Metrics: Accuracy, Precision, Recall, F1 (per-class & macro/weighted)
- Confusion matrix + training curves
- Sample predictions grid
- TorchScript export for deployment

##  Setup
```bash
pip install -r requirements.txt
```

##  Training
```bash
python train_cnn.py --dataset cifar10 --epochs 30 --batch-size 128
python train_cnn.py --dataset fashion_mnist --epochs 25
python train_cnn.py --dataset custom --data-root data/split_custom --img-size 128 --epochs 30
```

##  Outputs
After training, artifacts are saved in `outputs/`:
- `best_model.pth` – best weights by val loss
- `model_scripted.pt` – TorchScript for deployment
- `metrics.json` – accuracy, precision/recall/F1
- `confusion_matrix.png` – confusion matrix heatmap
- `history.png` + `history_loss.png` + `history_acc.png` – training curves
- `predictions_grid.png` – sample predictions
- `class_names.txt` – class labels

##  Inference
```bash
python inference_demo.py --image path/to/image.jpg
```

##  Report
Use `TECHNICAL_REPORT_TEMPLATE.md` to draft your 1–2 page summary. Fill in metrics from `outputs/metrics.json`.

---

💡 Author: Mohit Gupta
