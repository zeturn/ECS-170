# ECS 170 Stage 2 Report

## 1. Task Description
This stage focuses on multiclass classification using a pre-partitioned dataset. The training set and test set are fixed in two files:
- data/stage_2_data/train.csv (60,000 instances)
- data/stage_2_data/test.csv (10,000 instances)

Each row contains 785 comma-separated integers in the format:
label,feature1,feature2,...,feature784

The label is in {0,1,2,...,9}, and each feature is a grayscale pixel value.

## 2. Data Processing
### 2.1 Data Loading
We implemented a Stage 2 dataset loader that:
- Reads train.csv and test.csv directly (no random split and no cross validation)
- Uses the first column as label and the remaining 784 columns as features
- Supports multiclass classification directly

### 2.2 Preprocessing
We normalize image features from [0,255] to [0,1] by dividing all pixel values by 255.0.

### 2.3 Final Data Shape
- Training data: 60,000 x 784
- Test data: 10,000 x 784
- Number of classes: 10

## 3. Model and Training Setup
We used an MLP and ran two configurations: a baseline and an improved model.

### 3.1 Baseline Model
- Architecture: 784 -> 128 -> 64 -> 10
- Activation: ReLU
- Dropout: 0.1
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 20

### 3.2 Improved Model
- Architecture: 784 -> 256 -> 128 -> 64 -> 10
- Activation: ReLU
- Dropout: 0.2
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Optimizer: AdamW
- Learning rate: 5e-4
- Weight decay: 1e-4
- Batch size: 128
- Epochs: 30

The improved configuration was designed to satisfy the Stage 2 requirement for model tuning and performance improvement attempts.

## 4. Evaluation Metrics
Because this is a multiclass task, we report:
- Accuracy
- Macro Precision / Macro Recall / Macro F1
- Micro Precision / Micro Recall / Micro F1
- Weighted Precision / Weighted Recall / Weighted F1

## 5. Results
### 5.1 Quantitative Results
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9823 | 0.9821 | 0.9821 | 0.9821 | 0.9823 | 0.9823 |
| Improved | 0.9838 | 0.9838 | 0.9837 | 0.9837 | 0.9838 | 0.9838 |

### 5.2 Learning Curves
Generated files:
- result/stage_2_result/learning_curve_baseline.png
- result/stage_2_result/learning_curve_improved.png

Observed behavior:
- Both models converge quickly.
- The improved model trains longer and reaches higher training accuracy.
- The improved model also provides better test metrics, indicating better generalization.

## 6. Ablation and Improvement Discussion
Compared with baseline, the improved setup changed:
- More hidden layers / larger hidden dimensions
- Different optimizer (Adam -> AdamW)
- Different loss function (CrossEntropy -> Label Smoothing CrossEntropy)
- Longer training and stronger regularization

Performance gain:
- Accuracy: 0.9823 -> 0.9838 (+0.0015)
- Macro F1: 0.9821 -> 0.9837 (+0.0016)

These improvements are consistent with the expectation that increased capacity and better regularization can improve multiclass classification quality.

## 7. Final Conclusion
The Stage 2 pipeline was implemented with the required settings:
- Predefined train/test usage (no random split, no k-fold)
- Multiclass-aware evaluation metrics (macro/micro/weighted)
- Learning curve generation
- Hyperparameter tuning and model improvement attempts

Final best model: improved MLP
- Accuracy: 0.9838
- Macro F1: 0.9837

## 8. Limitations and Future Work
- MLP does not explicitly exploit 2D spatial locality of images.
- Future work can compare CNN-based architectures and learning-rate schedules.
- Additional tuning (early stopping, scheduler, broader hyperparameter sweep) may further improve performance.
