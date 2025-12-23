import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import time

def train_dcrnn_model(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    cost = 0
    for time, (X_batch, edges_index, edges_attr,
               y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()
        edges_index = edges_index.to(device)
        edges_attr = edges_attr.to(device).float()
        optimizer.zero_grad()
        y_pred = model(X_batch, edges_index, edges_attr)
        cost = cost + torch.mean((y_pred.squeeze() - y_batch)**2)
    cost = cost / (time + 1)

    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    return cost.cpu().item()


def evaluate_dcrnn_model(model, dataloader, criterion, device,):

    model.eval()

    total_loss = 0
    all_y_batch_np = []
    all_y_pred_np = []
    with torch.no_grad():
        for X_batch, edges_index, edges_attr, y_batch in dataloader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            edges_index = edges_index.to(device)
            edges_attr = edges_attr.to(device).float()
            y_pred = model(X_batch, edges_index, edges_attr)

            loss = criterion(y_pred.reshape(-1), y_batch)

            total_loss += loss.item()
            y_pred_np = y_pred.detach().cpu().reshape(-1).numpy()
            all_y_pred_np.append(y_pred_np)
            
            # Store true values
            y_batch_np = y_batch.detach().cpu().reshape(-1).numpy()
            all_y_batch_np.append(y_batch_np)
            avg_loss = total_loss / len(dataloader)
    final_y_pred_np = np.concatenate(all_y_pred_np)
    final_y_batch_np = np.concatenate(all_y_batch_np)
    tmp_final_y_pred_np = np.stack(all_y_pred_np)
    tmp_final_y_batch_np = np.stack(all_y_batch_np)
    print(f"Validation Losses: MSE : {mean_squared_error(final_y_batch_np, final_y_pred_np):.6f} MAE : {mean_absolute_error(final_y_batch_np, final_y_pred_np):.6f} RMSE : {root_mean_squared_error(final_y_batch_np, final_y_pred_np):.6f}")
    
    return tmp_final_y_pred_np, tmp_final_y_batch_np



def train_encoder_decoder_model(
        model,
        dataloader,
        criterion,
        optimizer,
        device):

    model.train()
    total_loss = 0

    for batch_idx, (X_batch, _, _, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_encoder_decoder_model(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0
    all_y_batch_np = []
    all_y_pred_np = []
    
    with torch.no_grad():
        for X_batch, _, _, y_batch in dataloader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            y_pred = model(X_batch)
            loss = criterion(y_pred.reshape(-1), y_batch.reshape(-1))
            # print(y_pred.shape, y_batch.shape)
            total_loss += loss.item()
            y_pred_np = y_pred.detach().cpu().reshape(-1).numpy()
            all_y_pred_np.append(y_pred_np)
            
            # Store true values
            y_batch_np = y_batch.detach().cpu().reshape(-1).numpy()
            all_y_batch_np.append(y_batch_np)
            avg_loss = total_loss / len(dataloader)
    final_y_pred_np = np.concatenate(all_y_pred_np)
    final_y_batch_np = np.concatenate(all_y_batch_np)
    tmp_final_y_pred_np = np.stack(all_y_pred_np)
    tmp_final_y_batch_np = np.stack(all_y_batch_np)
    print(f"Validation Losses: MSE : {mean_squared_error(final_y_batch_np, final_y_pred_np):.6f} MAE : {mean_absolute_error(final_y_batch_np, final_y_pred_np):.6f} RMSE : {root_mean_squared_error(final_y_batch_np, final_y_pred_np):.6f}")

    return tmp_final_y_pred_np, tmp_final_y_batch_np


def exponential_decay_fn(epoch):

    return 0.91 ** epoch


def training_loop(
        epochs,
        model,
        train_dataloader,
        mode="DCRNN"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if mode == "LSTM":
        reduce_lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=exponential_decay_fn
        )

        print(f"Starting training on {device} for {epochs} epochs...")
        start_time = time.time()
        
        training_losses = []
        test_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = train_encoder_decoder_model(
                model,
                train_dataloader,
                criterion,
                optimizer,
                device
            )
            print(f"Epoch {epoch}/{epochs} | Training Loss: {epoch_loss:.6f} ")

            reduce_lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            training_losses.append(epoch_loss)
        print(f"Training finished in {time.time() - start_time}s")
    elif mode == "DCRNN":
        print(f"Starting training on {device} for {epochs} epochs...")
        start_time = time.time()

        training_losses = []
        test_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = train_dcrnn_model(
                model,
                train_dataloader,
                criterion,
                optimizer,
                device,
            )
            print(f"Epoch {epoch}/{epochs} | Training Loss: {epoch_loss:.6f} ")

            training_losses.append(epoch_loss)
        print(f"Training finished in {time.time() - start_time}s")


def validation_loop(
        model,
        val_dataloader,
        mode="DCRNN"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    if mode == "LSTM":

        Y_pred, Y_true = evaluate_encoder_decoder_model(
            model, val_dataloader, criterion, device)
        return Y_pred, Y_true
    
    elif mode == "DCRNN":

        Y_pred, Y_true = evaluate_dcrnn_model(
            model, val_dataloader, criterion, device)
        return Y_pred, Y_true 
