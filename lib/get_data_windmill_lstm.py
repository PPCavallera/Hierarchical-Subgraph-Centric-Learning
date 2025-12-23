import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import scipy as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

def dinv(D):
    '''
    Computes inverse of diagonal matrix.
    '''
    eps = 1e-8
    d = np.diag(D)
    if np.any(d < eps):
        print('Warning: Ill-conditioned or singular matrix.')
    return np.diag(1 / d)


def sortEig(A, evs=5, which='LM'):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1]  # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])


def transitionMatrix(A, variant='rw'):
    '''
    Compute transition probability matrix.

    :param variant: Choose 'rw' (random-walk),
                            'fb' (forward-backward).
    '''
    D = np.diag(np.sum(A, 1))
    P = dinv(D) @ A

    if variant == 'rw':
        return P
    elif variant == 'fb':
        D_nu = np.diag(np.sum(P, 0))  # uniform density mapped forward
        Q = P @ dinv(D_nu) @ P.T
        return Q
    else:
        print('Unknown type.')


def spectralClustering(A, nc, variant='rw', mode=0):
    P = transitionMatrix(A, variant)
    d, V = sortEig(P, evs=nc, which='LR')

    kmeans = AgglomerativeClustering(n_clusters=nc).fit(np.real(V))
    c = kmeans.labels_
    return (d, V, c)


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

def create_adjacency_matrix(li, weights, max):
    A = [[0 for _ in range(max + 1)] for _ in range(max + 1)]
    for i, l in enumerate(li):
        A[l[0]][l[1]] = weights[i]
        A[l[1]][l[0]] = weights[i]
    return A


def reprocess_adjacency_matrix(A, c, cluster_id):
    li = []
    ind = []
    for i, v in enumerate(c):
        if v == cluster_id:
            ind.append(i)
            for j, e in enumerate(A[i]):
                if j >= i and c[j] == cluster_id and e == 1:
                    li.append([i, j])
    for l in li:
        l[0] = ind.index(l[0])
        l[1] = ind.index(l[1])
    return li


def get_raw_data():
    with open("./data/windmill_large.json") as f:
        data = json.load(f)
    di = {}
    for i in range(int(data['time_periods'])):
        di[str(i)] = data["block"][i]

    df = pd.DataFrame(di)

    max = 0
    for l in data["edges"]:
        if l[0] > max:
            max = l[0]
        if l[1] > max:
            max = l[1]
    A = create_adjacency_matrix(data["edges"], data['weights'], max)
    A = np.array(A)
    return df, A


def get_precise_cluster(df, A, nb_cluster, cluster_id, mode="HC"):
    if mode == "HC":
        model = AgglomerativeClustering(
            distance_threshold=None,
            n_clusters=nb_cluster,
            compute_full_tree=True,
            compute_distances=True,
            linkage="ward")

        model = model.fit(df.values)
        df = df.transpose()
        edges = reprocess_adjacency_matrix(A, model.labels_, cluster_id)
        return df[df.columns[model.labels_ == cluster_id]], edges
    elif mode == "SC":
        (d, V, c) = spectralClustering(A, nb_cluster)
        edges = reprocess_adjacency_matrix(A, c, cluster_id)
        df = df.transpose()
        return df[df.columns[c == cluster_id]], edges


def get_prepared_dataset(df, n_past, n_future, n_features):
    split = int(df.shape[0] * 0.80)
    train_df, test_df = df[0:split], df[split:]
    X_train, y_train = split_series(train_df.values, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_series(test_df.values, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    return X_train, y_train, X_test, y_test


class CustomDataset(Dataset):
    def __init__(self, x, edges_index, edges_attr, y):
        self.x = x
        self.edges_index = edges_index
        self.edges_attr = edges_attr
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.edges_index, self.edges_attr, self.y[idx]


def get_prepared_dataloader(X_train, y_train, X_test, y_test, edges):

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    edges_index = torch.from_numpy(np.asarray(edges)).reshape((2, len(edges)))
    edges_attr = torch.from_numpy(np.asarray([1 for _ in range(len(edges))]))

    train_dataset = CustomDataset(
        X_train_tensor,
        edges_index,
        edges_attr,
        y_train_tensor)
    test_dataset = CustomDataset(
        X_test_tensor,
        edges_index,
        edges_attr,
        y_test_tensor)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False)

    return train_dataloader, test_dataloader
