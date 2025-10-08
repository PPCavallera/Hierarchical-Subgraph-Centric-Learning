import pandas as pd
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering


def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        past, future = series[window_start:past_end,
                              :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


def normalizer(values):

    return (
        values - np.mean(values, axis=0)
    ) / np.std(values, axis=0)


def get_raw_data():
    with open("./data/wikivital_mathematics.json") as f:
        data = json.load(f)
    di = {}
    for i in range(int(data['time_periods'])):
        di[str(i)] = data[str(i)]['y']

    df = pd.DataFrame(di)

    vals = normalizer(df.values)
    df = pd.DataFrame(vals)
    return df


def get_precise_cluster(df, nb_cluster, cluster_id):
    model = AgglomerativeClustering(
        distance_threshold=None,
        n_clusters=nb_cluster,
        compute_full_tree=True,
        compute_distances=True,
        linkage="ward")

    model = model.fit(df.values)
    df = df.transpose()
    return df[df.columns[model.labels_ == cluster_id]]


def get_prepared_dataset(df, n_past, n_future, n_features):
    split = int(df.shape[0] * 0.80)
    train_df, test_df = df[1:split], df[split:]
    X_train, y_train = split_series(train_df.values, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_series(test_df.values, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    df = get_raw_data()
    df = get_precise_cluster(df, 3, 0)
    print(df)
    print(get_prepared_dataset(df, 4, 1, len(df.columns)))
