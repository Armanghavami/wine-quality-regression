# Wine Quality Regression with Deep Feedforward Neural Network

A PyTorch-based deep learning project that predicts the quality of red wine using physicochemical attributes. This repository includes data preprocessing, model definition, training with hyperparameter tuning using Weights & Biases (W&B), and evaluation.

## Project Overview

The goal of this project is to build a regression model that predicts the quality score of red wine samples from their physicochemical measurements (e.g., acidity, sugar content, pH). We employ a feedforward neural network implemented in PyTorch, enhanced with batch normalization, dropout, and L1 regularization. Hyperparameters are tuned via Bayesian optimization using Weights & Biases sweeps.

## Dataset

The dataset used is the kaggke Wine Quality dataset (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset), which includes 11 input features (e.g., acidity, alcohol content) and a quality score for red wine samples.

## Features

- **Data Preprocessing**: Loads and splits the dataset into training, validation, and test sets, with feature standardization using `StandardScaler`.
- **Model Architecture**: A custom neural network with four linear layers, batch normalization, leaky ReLU activations, and dropout for regularization.
- **Training**: Implements training with the Adam optimizer, MSE loss, L1 regularization, and a learning rate scheduler (`ReduceLROnPlateau`).
- **Evaluation**: Computes metrics including MSE, MAE, RMSE, and R² on the test set.
- **Hyperparameter Tuning**: Uses wandb sweeps to optimize layer sizes, dropout rates, learning rates, and batch sizes.
- **Reproducibility**: Sets a fixed seed (42) for consistent results.

## Data Preparation

1. **Load dataset** from CSV.
2. **Split** into training (60%), validation (20%), and test (20%) sets.
3. **Fit** `StandardScaler` on training features only.
4. **Transform** validation and test features using the same scaler.
5. **Convert** to PyTorch tensors and wrap in `DataLoader` objects.

## Model Architecture

```python
class WineRegressionModel(nn.Module):
    def __init__(self, input_dim, l1, l2, l3, output_dim, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.BatchNorm1d(l1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            nn.LeakyReLU(),
            nn.Linear(l2, l3),
            nn.BatchNorm1d(l3),
            nn.LeakyReLU(),
            nn.Linear(l3, output_dim)
        )
    def forward(self, x):
        return self.layers(x)
```

- **Input layer** size = number of features (11 physicochemical attributes).
- **Hidden layers** with configurable sizes (`l1`, `l2`, `l3`).
- **Output layer** size = 1 (predicted quality score).

## Training and Hyperparameter Tuning

- **Loss Function**: Mean Squared Error (MSE) + L1 regularization term.
- **Optimizer**: Adam with learning rate from sweep.
- **Scheduler**: `ReduceLROnPlateau` to lower LR when validation loss plateaus.
- **Early Stopping**: Stops if validation MSE does not improve for `patience` epochs.

### Sweep Configuration

```json
{
  "method": "bayes",
  "metric": {"name": "mse_epoch_valid", "goal": "minimize"},
  "parameters": {
    "l1_size": {"values": [32, 64, 128]},
    "l2_size": {"values": [64, 128, 256]},
    "l3_size": {"values": [32, 64]},
    "dropout_size": {"min": 0.2, "max": 0.5},
    "learning_rate": {"min": 1e-4, "max": 1e-2},
    "batch_size": {"values": [16, 32, 64]},
    "epochs": {"value": 50}
  }
}
```

## Evaluation Metrics

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

## Results of the first run before the hyperparameter tuning

| Metric | Value (Test Set) |
| ------ | ---------------- |
| MSE    | 0.44             |
| MAE    | 0.52             |
| RMSE   | 0.67             |
| R²     | 0.26             |

## The best results after hyperparameter tuning using Wandb sweep

| Metric | Value (Test Set) |
| ------ | ---------------- |
| MSE    | 0.40             |
| MAE    | 0.49             |
| RMSE   | 0.63             |
| R²     | 0.33             |


## Model Performance Analysis

We observe a decrease in the MSE to a threshold of about 0.40, which the model has not been able to break lower, even with 30 configurations of hyperparameter tuning using Weights & Biases (wandb). This suggests one or many of the following reasons as the cause:

1. **Lack of Model Complexity and Depth**  
   There might be insufficient model complexity and depth. Perhaps adding a few layers to the architecture will help. The reason I did not do this in this case was my lack of current computational power, as with more layers, there would be more models to train and test for hyperparameter tuning.

2. **Insufficient Number of Samples**  
   There might be a lack of samples (there are only 1600 samples, and we know that neural networks require a huge amount of data to learn the features properly and avoid overfitting).

3. **Early Stopping Patience**  
   I can increase the patience for early stopping to give more chances to the learning rate function to lower itself and maybe improve the learning and find a lower optimal gradient.

4. **L1 Regularization Strength**  
   We can increase the `l1_lambda` and thereby increase the effect of L1 regularization and further decrease overfitting. The MSE of the train set of the best model for the last epoch was 0.33, which is a 0.14 difference from the test set, indicating some levels of overfitting.




# Next Steps
To address these limitations:

1. Experimenting with deeper architectures when computational resources are available.
2. Exploring data augmentation or acquiring a larger dataset.
3. Adjusting the early stopping patience to allow more training epochs.
4. Tuning the l1_lambda parameter to balance regularization and model performance.
