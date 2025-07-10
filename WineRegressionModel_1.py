import pandas as pd 
import torch 
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
import random
import numpy as np




# Set seeds for accurate model comparison and hyperparameter tuning

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# data loading and preprocessing

def prepare_data(path,test_size=0.2,valid_size=0.2):
    """
    1. Load CSV data
    2. Split into train, validation, test
    3. Fit scaler on training features ONLY
    4. Transform train/val/test sets
    5. Convert to PyTorch Tensors and DataLoaders
    """
    wine_dataset = pd.read_csv(path)


    x = wine_dataset.drop(columns="quality")
    y = wine_dataset["quality"]


    x_train,x_temp,y_train,y_temp = train_test_split(x,y,shuffle=True,train_size=(1-(test_size+valid_size)),random_state=42)
    x_test,x_valid,y_test,y_valid = train_test_split(x_temp,y_temp,train_size=(1-valid_size),random_state=42)


    z_score = StandardScaler()
    x_train_norm = z_score.fit_transform(x_train)
    x_test_norm = z_score.transform(x_test)
    x_valid_norm = z_score.transform(x_valid)

    
    x_trian_norm_tensor = torch.tensor(x_train_norm,dtype=torch.float32)
    x_test_norm_tensor = torch.tensor(x_test_norm,dtype=torch.float32)
    x_valid_norm_tensor = torch.tensor(x_valid_norm,dtype=torch.float32)


    y_train_norm = torch.tensor(y_train.values,dtype=torch.float32)
    y_test_norm = torch.tensor(y_test.values,dtype=torch.float32)
    y_valid_norm = torch.tensor(y_valid.values,dtype=torch.float32)


    train_ds = TensorDataset(x_trian_norm_tensor,y_train_norm)
    test_ds = TensorDataset(x_test_norm_tensor,y_test_norm)
    valid_ds = TensorDataset(x_valid_norm_tensor,y_valid_norm)


    train_dl = DataLoader(train_ds,batch_size=wandb.config.batch_size)
    test_dl = DataLoader(test_ds,batch_size=wandb.config.batch_size)
    valid_dl = DataLoader(valid_ds,batch_size=wandb.config.batch_size)


    return train_dl,test_dl,valid_dl,x_trian_norm_tensor.size(1)






# model definition

class WineRegressionModel(nn.Module):
    def __init__(self,l0,l1,l2,l3,l4,dropout_size):
        super().__init__()

        self.l1 = nn.Linear(l0,l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.dropout1 = nn.Dropout(dropout_size)

        self.l2 = nn.Linear(l1,l2)
        self.bn2 = nn.BatchNorm1d(l2)

        self.l3 = nn.Linear(l2,l3)
        self.bn3 = nn.BatchNorm1d(l3)


        self.l4 = nn.Linear(l3,l4)

    def forward(self,x):

        x = self.l1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.l4(x)
        return x 
    





# training and evaluation with l1 regularization and logging the information in wandb 


def train(model, train_dl, valid_dl, loss_fn, optimizer, scheduler, epochs, l1_lambda=1e-5,early_stopping_patience=10, early_stopping_delta=0.0):
    mse_train =[]
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None


    for epoch in range(epochs):
        totall = 0 
        loss_batch_train = 0
        model.train()


        for data_train , target_train in train_dl:
            data_train = data_train.to(device)
            target_train = target_train.to(device)
            optimizer.zero_grad()

            y_hat_train = model(data_train)
            loss_curent_train = loss_fn(y_hat_train.squeeze(),target_train)


            l1_lambda = 1e-5  
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss_curent_train = loss_curent_train + l1_lambda * l1_norm
            loss_batch_train += loss_curent_train * data_train.size(0)
            totall+= data_train.size(0)


            loss_curent_train.backward()
            optimizer.step()


        mse_epoch_train = loss_batch_train/totall
        mse_train.append(mse_epoch_train.item())
        

        mse_valid = []
        mse_batch_valid = 0 
        totall_valid = 0 
        y_true_valid = []
        y_pred_valid = []

        for data_valid , target_valid in valid_dl:

            data_valid = data_valid.to(device)
            target_valid = target_valid.to(device)
            model.eval()


            y_hat_valid = model(data_valid)
            loss_curent_valid = loss_fn(y_hat_valid.squeeze(),target_valid)

            mse_batch_valid += loss_curent_valid * data_valid.size(0)
            totall_valid += data_valid.size(0)

            y_true_valid.extend(target_valid.numpy().tolist())        
            y_pred_valid.extend(y_hat_valid.squeeze().detach().numpy().tolist())
            


        mae_valid = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse_valid = root_mean_squared_error(y_true_valid, y_pred_valid)
        r2_valid = r2_score(y_true_valid, y_pred_valid)
        mse_epoch_valid = mse_batch_valid/totall_valid




        # early stopping check
        if mse_epoch_valid < best_val_loss - early_stopping_delta:
            best_val_loss = mse_epoch_valid
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break



        mse_valid.append(mse_epoch_valid.item())
        print(f"Epoch {epoch+1} | Train MSE: {mse_epoch_train:.4f} | Valid MSE: {mse_epoch_valid:.4f}")

        scheduler.step(loss_curent_valid)

        wandb.log({
            "epoch":epoch,
            "mse_epoch_train":mse_epoch_train,
            "mse_epoch_valid":mse_epoch_valid,
            "Lr":optimizer.param_groups[0]['lr'],
            "mae_epoch_valid": mae_valid,
            "rmse_epoch_valid": rmse_valid,
            "r2_epoch_valid": r2_valid,

        

        })



def evaluate(model, test_dl, loss_fn):



    mse_test = []
    mse_batch_test = 0 
    totall_test = 0 
    y_true_test = []
    y_pred_test = []

    for data_test , target_test in test_dl:
        data_test = data_test.to(device)
        target_test = target_test.to(device)
        model.eval()

        y_hat_test = model(data_test)
        loss_curent_test = loss_fn(y_hat_test.squeeze(),target_test)

        mse_batch_test += loss_curent_test * data_test.size(0)
        totall_test += data_test.size(0)

        y_true_test.extend(target_test.numpy().tolist())
        y_pred_test.extend(y_hat_test.squeeze().detach().numpy().tolist())


    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)

    print("mae_test: ", mae_test)
    print("rmse_test:", rmse_test)
    print("r2_test:  ", r2_test)

            
    mse_epoch_test = mse_batch_test/totall_test
    mse_test.append(mse_epoch_test.item())
    print("mse_test:  ",mse_epoch_test.item())

    wandb.log({
        "mse_epoch_test":mse_epoch_test,
        "mae_epoch_test": mae_test,
        "rmse_epoch_test": rmse_test,
        "r2_epoch_test": r2_test,


    })







sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'mse_epoch_valid', 'goal': 'minimize'},
    'parameters': {
        'l1_size': {'values': [32, 64, 128]},
        'l2_size': {'values': [64, 128, 256]},
        'l3_size': {'values': [32, 64]},
        'dropout_size': {'min': 0.2, 'max': 0.5},
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'value': 50}
    }
}


def sweep_main():
    wandb.init(project="wine_regression_sweep", config=sweep_config)
    train_dl, test_dl, valid_dl, input_dim = prepare_data(
        "file_path", 
        test_size=0.2, valid_size=0.2
    )

    model = WineRegressionModel(
        l0=input_dim,
        l1=wandb.config.l1_size,
        l2=wandb.config.l2_size,
        l3=wandb.config.l3_size,
        l4=1,
        dropout_size=wandb.config.dropout_size
    )
    model.to(device)


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train(model, train_dl, valid_dl, loss_fn, optimizer, scheduler, epochs=wandb.config.epochs)
    evaluate(model, test_dl, loss_fn)

    torch.save(model,"path.pth")
    print("Model weights saved")




if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="wine_regression_sweep_7")
    wandb.agent(sweep_id, function=sweep_main,count=1)



