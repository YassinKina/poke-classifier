# Pokémon Image Classifier: Dynamic CNN with HPO

A high-performance deep learning pipeline designed to classify the original 150 Pokémon species. This project implements a custom **DynamicCNN** architecture that allows for automated architectural searches, combined with a rigorous Hyperparameter Optimization (HPO) workflow.



[Image of a convolutional neural network architecture for image classification]


## 📊 Performance Summary
* **Top-1 Accuracy:** `67.86%` (Exact species match)
* **Top-5 Accuracy:** `89.40%` (Correct species in the top 5 candidates)
* **Optimization:** 20-trial study using Bayesian TPE Sampling and Median Pruning.

---

## 🛠 Features

### 1. Dynamic Architecture
The `DynamicCNN` is a flexible PyTorch implementation that adapts to configuration-driven depth and width:
- **Variable Depth:** Supports dynamic `n_layers` configuration.
- **Adaptive Width:** Adjusts `n_filters` and `fc_size` based on HPO suggestions.
- **Regularization:** Integrated Dropout, Batch Normalization, and Weight Decay.

### 2. Automated HPO Workflow
Leveraging **Optuna** and **Hydra**, the training pipeline explores a multi-dimensional search space:
- **Optimizer Params:** Learning Rate ($1^{-5}$ to $1^{-4}$), Weight Decay ($1^{-6}$ to $1^{-2}$).
- **Regularization:** Dropout rates and Label Smoothing (up to 0.2).
- **Architecture:** Layer counts and fully-connected layer dimensions.
- **Early Stopping:** `MedianPruner` terminates underperforming trials to optimize compute resources.



### 3. Professional Experiment Tracking
- **Weights & Biases:** Real-time logging of training/validation loss, Top-1 accuracy, Top-5 accuracy, and learning rate curves.
- **Hydra:** Version-controlled configuration management for reproducible experiments.

---

## 📁 Project Structure
```text
.
├── config/             # Hydra YAML configurations (hpo vs. train)
├── data/               # Pokémon dataset (Cleaned & Preprocessed)
├── models/             # Saved checkpoints (state_dicts + metadata)
├── src/
│   ├── dataset.py      # Custom Dataset with Pokemon-specific stats
│   ├── model.py        # DynamicCNN architecture
│   ├── engine.py       # Training, Val, and Top-K Eval logic
│   └── utils.py        # Stats calculation and W&B initialization
├── hpo.py              # Optuna optimization entry point
├── eval.py             # Script for final test-set evaluation
└── predict.py          # CLI tool for single-image inference 
```

## 🚀 Getting Started

### 1. Installation
 ```pip install requirements.txt```

### 2. Run Hyperparameter Optimization
Launches a new study (of 20 trials) with Bayesian search
 ```python hpo.py``

### 3. Run Final Evaluation
Load the best weights from the ```models/ ```directory and evaluate on the hold-out test set:
``` python eval.py ```

### 4. Single Image Inference


## 🧪  Data Normalization
This project uses custom-calculated channel-wise statistics to account for the unique color distribution of Pokémon art 
(higher brightness and saturation compared to natural images) rather than standard ImageNet defaults:
* Mean: [0.5863, 0.5675, 0.5337]
* Std: [0.3464, 0.3312, 0.3421]


## Data Limitaions
Very few training data were pictures of pokemon cards. As a result, the model struggles to correctly classify
the input when given a pokemon card image.


