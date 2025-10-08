import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def train_models(
        X_train,
        y_train,
        X_test,
        y_test,
        model_1,
        model_2,
        bs=8,
        nb_epochs=100):
    # Training the models
    nb_epochs = nb_epochs
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda x: 1e-3 * 0.90 ** x)
    # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                          patience=5)
    print("--- FIRST MODEL ---")
    model_1.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.Huber())
    history_1 = model_1.fit(
        X_train,
        y_train,
        epochs=nb_epochs,
        validation_data=(
            X_test,
            y_test),
        batch_size=bs,
        verbose=2,
        callbacks=[reduce_lr])
    print("--- SECOND MODEL ---")
    model_2.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.Huber())
    history_2 = model_2.fit(
        X_train,
        y_train,
        epochs=nb_epochs,
        validation_data=(
            X_test,
            y_test),
        batch_size=bs,
        verbose=2,
        callbacks=[reduce_lr])

    return model_1, model_2, history_1, history_2


def save_loss_graphs(history_1, history_2, loss_name, nb_epochs=100):
    loss_e1d1 = history_1.history["loss"]
    val_loss_e1d1 = history_1.history["val_loss"]

    loss_e2d2 = history_2.history["loss"]
    val_loss_e2d2 = history_2.history["val_loss"]

    # Plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, nb_epochs + 1)

    # Model e1d1
    plt.plot(epochs, loss_e1d1, label="Train Loss e1d1")
    plt.plot(epochs, val_loss_e1d1, label="Val Loss e1d1")

    # Model e2d2
    plt.plot(epochs, loss_e2d2, label="Train Loss e2d2")
    plt.plot(epochs, val_loss_e2d2, label="Val Loss e2d2")

    plt.xlabel("Epochs")
    plt.ylabel("Loss (Huber)")
    plt.title("Training & Validation Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_name + ".png")
    loss_df = pd.DataFrame({"loss_m1": loss_e1d1,
                            "val_loss_m1": val_loss_e1d1,
                            "loss_m2": loss_e2d2,
                            "val_loss_m2": val_loss_e2d2})
    loss_df.to_csv(loss_name + ".csv", index=False)


def get_test_loss(X_test, y_test, model_1, model_2, name):
    pred_e1d1 = model_1.predict(X_test)
    pred_e2d2 = model_2.predict(X_test)
    print(y_test.shape)
    loss_1, loss_2 = 0, 0
    for index in range(y_test.shape[2]):

        for j in range(y_test.shape[1]):
            print("Day ", j, ":")
            mse_1, mse_2 = mean_squared_error(y_test[:, j - 1, index], pred_e1d1[:, j - 1, index]), mean_squared_error(y_test[:, j - 1, index], pred_e2d2[:, j - 1, index])
            print("MSE-M1 : ", mse_1, end=", ")
            print("MSE-M2 : ", mse_2)
            loss_1 += mse_1
            loss_2 += mse_2
    print(f'GLOBAL PRECISION : MSE MODEL 1 : {loss_1 /(y_test.shape[0] *y_test.shape[2]) if (y_test.shape[0]*y_test.shape[2])>0 else loss_1} --- MSE MODEL 2 : {loss_2 /(y_test.shape[0] * y_test.shape[2]) if (y_test.shape[0]*y_test.shape[2])>0 else loss_2}')
    
    for feat_idx in range(pred_e1d1.shape[2]):
    
        truth_series = y_test[:min(100, len(y_test)), 0, feat_idx]   # take the "day+1" target from each window

    # predictions: take the same "day+1" forecast from each model
        pred_series_e1d1 = pred_e1d1[:min(100, len(y_test)), 0, feat_idx]
        pred_series_e2d2 = pred_e2d2[:min(100, len(y_test)), 0, feat_idx]

        # align with actual test dates (skip first n_past days used as input)
        test_dates = [i for i in range(min(100, len(pred_e1d1)))]

        plt.figure(figsize=(12,4))
        plt.plot(test_dates, truth_series, label="True", linewidth=2)
        plt.plot(test_dates, pred_series_e1d1, label="Pred LSTM 1 Hidden Layer", linestyle="--")
        plt.plot(test_dates, pred_series_e2d2, label="Pred LSTM 2 Hidden Layers", linestyle="--")
        #plt.title(f"Predictions vs Truth for feature")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(name+"_vertice"+str(feat_idx)+".png")


