# ECS 170 Stage 2 Report Template

Use this template to write your Stage 2 report. Keep the full report within 5 pages.

## 1. Task Description
- Dataset and task type: handwritten digit recognition (multiclass classification, 10 classes).
- Given split: predefined train set (`train.csv`) and test set (`test.csv`).
- Objective: train MLP on training set and evaluate performance on test set.

## 2. Data Processing
- Data file format: each row has 785 comma-separated integers in the order `label, feature1, ..., feature784`.
- Number of instances: 60,000 training samples and 10,000 testing samples.
- Number of features/classes: 784 input features and 10 output classes (`0` to `9`).
- Preprocessing: parse label from the first column and normalize pixel features from `[0, 255]` to `[0, 1]` by dividing by `255.0`.

## 3. Model Design
- Baseline MLP architecture: `784 -> 128 -> 64 -> 10` with ReLU and dropout.
- Improved MLP architecture: `784 -> 256 -> 128 -> 64 -> 10` with ReLU and dropout.
- Loss functions:
	- Baseline: cross entropy.
	- Improved: cross entropy with label smoothing (`0.1`).
- Optimizers:
	- Baseline: Adam.
	- Improved: AdamW.
- Main hyperparameters:
	- Baseline: epoch `20`, learning rate `1e-3`, batch size `128`, dropout `0.1`, weight decay `0`.
	- Improved: epoch `30`, learning rate `5e-4`, batch size `128`, dropout `0.2`, weight decay `1e-4`.

## 4. Evaluation Metrics
Report multiclass metrics:
- Accuracy
- Macro Precision / Recall / F1
- Micro Precision / Recall / F1
- Weighted Precision / Recall / F1

Measured results:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9823 | 0.9821 | 0.9821 | 0.9821 | 0.9823 | 0.9823 |
| Improved | 0.9838 | 0.9838 | 0.9837 | 0.9837 | 0.9838 | 0.9838 |

## 5. Learning Curve
- Baseline curve: `result/stage_2_result/learning_curve_baseline.png`
- Improved curve: `result/stage_2_result/learning_curve_improved.png`
- Convergence behavior:
	- Baseline quickly converges and reaches high training accuracy around epoch 10 to 20.
	- Improved model converges slightly slower but reaches higher final training accuracy and better test metrics.

## 6. Tuning and Improvements
Describe at least a few model improvement attempts, such as:
- Deeper network (more hidden layers)
- Different loss function
- Different optimizer
- Other training settings

For each attempt, report:
- Attempt A (baseline):
	- Configuration: `784-128-64-10`, Adam, cross entropy.
	- Key result: macro F1 `0.9821`.
- Attempt B (improved):
	- Configuration: `784-256-128-64-10`, AdamW, label smoothing cross entropy, longer training.
	- Key result: macro F1 `0.9837`.
- Comparison:
	- Improved model outperforms baseline by about `+0.0016` macro F1 and `+0.0015` accuracy.

## 7. Final Results and Analysis
- Best configuration: improved setting (`784-256-128-64-10`, AdamW, label smoothing).
- Final test metrics:
	- Accuracy: `0.9838`
	- Macro Precision / Recall / F1: `0.9838 / 0.9837 / 0.9837`
	- Micro Precision / Recall / F1: `0.9838 / 0.9838 / 0.9838`
	- Weighted Precision / Recall / F1: `0.9838 / 0.9838 / 0.9838`
- Why this configuration works better:
	- Larger model capacity captures more complex feature patterns.
	- AdamW + weight decay improves generalization.
	- Label smoothing reduces overconfidence and helps multiclass robustness.
- Limitations:
	- Pure MLP does not leverage 2D spatial structure of images, so performance can still be improved by CNN-like models.

## 8. Conclusion
- Summary: Stage 2 pipeline successfully trains and evaluates multiclass MLP on predefined train/test data and reaches strong test performance (>98% accuracy).
- Potential future improvements:
	- Introduce learning rate scheduling and early stopping.
	- Tune hidden width/depth and regularization more systematically.
	- Compare against CNN architectures for further gains.
