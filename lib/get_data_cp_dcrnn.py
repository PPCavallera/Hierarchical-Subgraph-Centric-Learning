import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scipy as sp
import json
from sklearn.cluster import AgglomerativeClustering


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


def create_adjacency_matrix(li, max):
    A = [[0 for _ in range(max + 1)] for _ in range(max + 1)]
    for l in li:
        A[l[0]][l[1]] = 1
        A[l[1]][l[0]] = 1
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
    print(li)
    for l in li:
        l[0] = ind.index(l[0])
        l[1] = ind.index(l[1])
    print(li)
    return li


def get_raw_data():
    with open("./data/chickenpox.json") as f:
        data = json.load(f)
    di = {}
    for i in range(len(data['FX'])):
        di[str(i)] = data["FX"][i]

    df = pd.DataFrame(di)

    max = 0
    for l in data["edges"]:
        if l[0] > max:
            max = l[0]
        if l[1] > max:
            max = l[1]
    A = create_adjacency_matrix(data["edges"], max)
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
        return model.labels_
    elif mode == "SC":
        (d, V, c) = spectralClustering(A, nb_cluster)
        return c


def get_processed_data(res, nb_past, cluster_index):
    with open("./data/chickenpox.json") as f:
        data = json.load(f)
    stacked_target = np.array(data["FX"])
    X = np.array([
        stacked_target[i: i + nb_past, :].T
        for i in range(stacked_target.shape[0] - nb_past)
    ])
    Y = np.array([
        stacked_target[i + nb_past, :].T
        for i in range(stacked_target.shape[0] - nb_past)
    ])
    threshold = round(len(X) * 0.8)
    X_train, X_test = X[:threshold], X[threshold:]
    y_train, y_test = Y[:threshold], Y[threshold:]
    if res is not None:
        X_train = X_train[:, np.where(res == cluster_index)[0], :]
        y_train = y_train[:, np.where(res == cluster_index)[0]]
        X_test = X_test[:, np.where(res == cluster_index)[0], :]
        y_test = y_test[:, np.where(res == cluster_index)[0]]

    edges = (np.asarray(data["edges"])).reshape((2, 102))
    if res is not None:
        li1 = []
        li2 = []
        vals = np.where(res == cluster_index)[0]
        for i in range(len(edges[0])):
            if edges[0][i] in vals and edges[1][i] in vals:
                li1.append(edges[0][i])
                li2.append(edges[1][i])
        for i in range(len(li1)):
            li1[i] = np.where(vals == li1[i])[0][0]
            li2[i] = np.where(vals == li2[i])[0][0]

        edges_index = torch.from_numpy(np.stack([li1, li2]))
    else:
        edges_index = torch.from_numpy(edges)
    edges_attr = torch.from_numpy(np.asarray(
        [1 for _ in range(edges_index.shape[1])]))
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, edges_index, edges_attr


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class.
    """

    def __init__(self, x, edges_index, edges_attr, y):
        # 1. Initialization
        # Load your data here (e.g., file paths, database connection, pandas DataFrame)
        # and store any necessary attributes like transforms.
        self.x = x
        self.edges_index = edges_index
        self.edges_attr = edges_attr
        self.y = y

    def __len__(self):
        # 2. Return the total number of samples in the dataset.
        return len(self.x)

    def __getitem__(self, idx):
        # 3. Retrieve one sample (data and label) at the given index.
        # This is called by the DataLoader.

        # Load sample from storage

        # Apply transforms if provided

        # Return the sample as a tuple of tensors
        return self.x[idx], self.edges_index, self.edges_attr, self.y[idx]


def get_prepared_dataloader(
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        edges_index,
        edges_attr):
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
        batch_size=None,
        shuffle=False,
        drop_last=False)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        shuffle=False,
        drop_last=False)
    return train_dataloader, test_dataloader
