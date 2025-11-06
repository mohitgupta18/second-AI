# CNN Image Classifier — Technical Report (1–2 pages)

## Objective
Design, train, and evaluate a multi-class image classifier using a CNN, demonstrating data preprocessing, architecture choices, optimization, and deployment readiness.

## Dataset
- Dataset Used: <CIFAR-10 / Fashion-MNIST / Custom>
- Split: Train 70% / Val 15% / Test 15%
- For CIFAR-10 and Fashion-MNIST, validation is carved out of the official train set; official test set used as-is.

## Data Preprocessing & Augmentation
- Resize to 32×32 (CIFAR-10/Fashion-MNIST) or 128×128 (custom).
- Normalize with ImageNet stats (grayscale uses single-channel mean/std).
- Training augmentation: RandomHorizontalFlip, RandomCrop (padding=4).
- Validation/Test: deterministic transforms, no augmentation.

## Model Architecture (PyTorch)
- 3 convolutional blocks; each block has:
  - Two 3×3 Conv layers → BatchNorm → ReLU → MaxPool → Dropout(0.3).
- Global Average Pooling (AdaptiveAvgPool2d) to reduce spatial dimension.
- Classifier: Linear(128→128) → ReLU → Dropout(0.3) → Linear(128→C).
- Rationale: BN stabilizes training; Dropout combats overfitting; GAP reduces parameters.

## Training & Optimization
- Loss: CrossEntropyLoss (multi-class).
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4).
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 3).
- Early Stopping: patience=7 on val loss.
- Mixed precision (AMP) on GPU when available.
- Reproducibility: seeds fixed, deterministic cuDNN where possible.

## Evaluation & Visualizations
- Validation during training: accuracy & loss.
- Final test set metrics:
  - Accuracy: <FINAL_TEST_ACC>
  - Macro F1: <MACRO_F1>, Weighted F1: <WEIGHTED_F1>
- Confusion matrix (`confusion_matrix.png`).
- Training curves (`history.png`).
- Sample predictions grid (`predictions_grid.png`).

## Results & Observations
- Observed confusions: <TOP_CONFUSIONS> (from confusion matrix).
- Curve behavior: <UNDERFITTING / GOOD FIT / OVERFITTING>.
- LR reductions at epochs: <EPOCHS_WHERE_LR_DROPPED>.

## Challenges & Improvements
- Overfitting mitigated with Dropout, augmentation, early stopping.
- If class imbalance: consider class-weighted loss or focal loss.
- Potential upgrades:
  1) Transfer learning (e.g., ResNet18/MobileNetV3).
  2) Stronger augmentation (RandAugment, MixUp/CutMix).
  3) EMA of weights.
  4) Temperature scaling for calibration.

## Deployment Readiness
- Best weights: `outputs/best_model.pth`.
- TorchScript: `outputs/model_scripted.pt`.
- Class names: `outputs/class_names.txt`.
- Metrics & plots saved for auditability.

## Conclusion
The pipeline satisfies requirements with robust training, comprehensive evaluation, and deployable artifacts. Further gains likely with transfer learning and richer augmentation.
