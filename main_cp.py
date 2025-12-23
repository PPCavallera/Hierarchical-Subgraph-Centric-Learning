from lib.get_data_cp_lstm import get_prepared_dataloader as get_prepared_dataloader_lstm, get_precise_cluster as get_precise_cluster_lstm, get_raw_data as get_raw_data_lstm, get_prepared_dataset as get_prepared_dataset_lstm
from lib.get_data_cp_dcrnn import get_prepared_dataloader as get_prepared_dataloader_dcrnn, get_raw_data as get_raw_data_dcrnn, get_precise_cluster  as get_precise_cluster_dcrnn, get_processed_data as get_processed_data_dcrnn

from lib.models import RecurrentGCN, EncoderDecoder, TwoLayerEncoderDecoder
from lib.training import training_loop, validation_loop
import matplotlib.pyplot as plt
import time
import numpy as np

NB_CLUSTER = 3
NB_PAST_VALUE = 4
NB_FUTURE_VALUE = 1

df, A = get_raw_data_lstm()
li_pred = []
li_true = []
li_columns = []
for c in range(NB_CLUSTER):
    df_c, edges = get_precise_cluster_lstm(df, A, NB_CLUSTER, c, mode="SC")
    for column in df_c.columns:
        li_columns.append(column)
    print(
        f"Cluster {c} : Number of elements : {len(df_c.columns)} out of {len(df.transpose().columns)} elements")
    X_train, y_train, X_test, y_test = get_prepared_dataset_lstm(
        df_c, NB_PAST_VALUE, NB_FUTURE_VALUE, len(df_c.columns))
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataloader, test_dataloader = get_prepared_dataloader_lstm(
        X_train, y_train, X_test, y_test, edges)
    model = EncoderDecoder(
        n_features=len(
            df_c.columns),
        n_past=NB_PAST_VALUE,
        n_future=NB_FUTURE_VALUE,
        units=128)
    training_loop(50, model, train_dataloader, mode="LSTM")
    y_pred, y_true = validation_loop(model, test_dataloader, mode="LSTM")
    li_pred.append(y_pred)
    li_true.append(y_true)
    # print(y_pred.shape)


SC1_pred = np.hstack(li_pred)[:,np.argsort(li_columns)]
Y_true = np.hstack(li_true)[:,np.argsort(li_columns)]


li_pred = []
li_true = []
li_columns = []
for c in range(NB_CLUSTER):
    df_c, edges = get_precise_cluster_lstm(df, A, NB_CLUSTER, c, mode="SC")
    for column in df_c.columns:
        li_columns.append(column)
    print(
        f"Cluster {c} : Number of elements : {len(df_c.columns)} out of {len(df.transpose().columns)} elements")
    X_train, y_train, X_test, y_test = get_prepared_dataset_lstm(
        df_c, NB_PAST_VALUE, NB_FUTURE_VALUE, len(df_c.columns))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataloader, test_dataloader = get_prepared_dataloader_lstm(
        X_train, y_train, X_test, y_test, edges)
    model = TwoLayerEncoderDecoder(
        n_features=len(
            df_c.columns),
        n_past=NB_PAST_VALUE,
        n_future=NB_FUTURE_VALUE,
        units=128)
    training_loop(50, model, train_dataloader, mode="LSTM")
    y_pred, y_true = validation_loop(model, test_dataloader, mode="LSTM")
    li_pred.append(y_pred)
    
SC2_pred = np.hstack(li_pred)[:,np.argsort(li_columns)]


li_pred = []
li_true = []
li_columns = []
for c in range(NB_CLUSTER):
    df_c, edges = get_precise_cluster_lstm(df, A, NB_CLUSTER, c, mode="HC")
    for column in df_c.columns:
        li_columns.append(column)
    print(
        f"Cluster {c} : Number of elements : {len(df_c.columns)} out of {len(df.transpose().columns)} elements")
    X_train, y_train, X_test, y_test = get_prepared_dataset_lstm(
        df_c, NB_PAST_VALUE, NB_FUTURE_VALUE, len(df_c.columns))
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataloader, test_dataloader = get_prepared_dataloader_lstm(
        X_train, y_train, X_test, y_test, edges)
    model = EncoderDecoder(
        n_features=len(
            df_c.columns),
        n_past=NB_PAST_VALUE,
        n_future=NB_FUTURE_VALUE,
        units=128)
    training_loop(50, model, train_dataloader, mode="LSTM")
    y_pred, y_true = validation_loop(model, test_dataloader, mode="LSTM")
    li_pred.append(y_pred)
    # y_true.append(y_true)
    # print(y_pred.shape)


HC1_pred = np.hstack(li_pred)[:,np.argsort(li_columns)]


li_pred = []
li_true = []
li_columns = []
for c in range(NB_CLUSTER):
    df_c, edges = get_precise_cluster_lstm(df, A, NB_CLUSTER, c, mode="HC")
    for column in df_c.columns:
        li_columns.append(column)
    print(
        f"Cluster {c} : Number of elements : {len(df_c.columns)} out of {len(df.transpose().columns)} elements")
    X_train, y_train, X_test, y_test = get_prepared_dataset_lstm(
        df_c, NB_PAST_VALUE, NB_FUTURE_VALUE, len(df_c.columns))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataloader, test_dataloader = get_prepared_dataloader_lstm(
        X_train, y_train, X_test, y_test, edges)
    model = TwoLayerEncoderDecoder(
        n_features=len(
            df_c.columns),
        n_past=NB_PAST_VALUE,
        n_future=NB_FUTURE_VALUE,
        units=128)
    training_loop(50, model, train_dataloader, mode="LSTM")
    y_pred, y_true = validation_loop(model, test_dataloader, mode="LSTM")
    li_pred.append(y_pred)
    
HC2_pred = np.hstack(li_pred)[:,np.argsort(li_columns)]


# res = get_precise_cluster_dcrnn(df, A, 1, 0, mode="HC")
# print(f"Cluster {c} : Number of elements : {len(np.where(res == c)[0])} out of {len(df.transpose().columns)} elements")
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, edges_index, edges_attr = get_processed_data_dcrnn(
    res=None, nb_past=NB_PAST_VALUE, cluster_index=c)
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
train_dataloader, test_dataloader = get_prepared_dataloader_dcrnn(
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, edges_index, edges_attr)
model = RecurrentGCN(NB_PAST_VALUE)
training_loop(50, model, train_dataloader, mode="DCRNN")
DCRNN_pred, Y_true = validation_loop(model, test_dataloader)

print(Y_true.shape, SC1_pred.shape, SC2_pred.shape, HC1_pred.shape, HC2_pred.shape, DCRNN_pred.shape)
for i in range(len(Y_true[0])):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(list(range(100)), Y_true[:100,i], label="Ground Truth", color='blue', linewidth=2)
    ax.plot(list(range(100)), SC1_pred[:100,i], label='SC-HC 1 Layer', color='red', linestyle='--', linewidth=2)
    ax.plot(list(range(100)), SC2_pred[:100,i], label='SC-HC 2 Layer2', color='green', linestyle=':', linewidth=2)
    ax.plot(list(range(100)), HC1_pred[:100,i], label='HC 1 Layer', color='purple', linestyle='-.', linewidth=2)
    ax.plot(list(range(100)), HC2_pred[:100,i], label='HC 1 Layer', color='orange', linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(list(range(100)), DCRNN_pred[:100,i], label='DCRNN', color='brown', ls='--', lw=2, alpha=0.7)
    ax.set_title('CP - Model comparison for node ' + str(i+1), fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig('CP_node'+str(i+1)+'.png')