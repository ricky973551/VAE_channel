from torch.utils.data import Dataset
import numpy as np
import torch
import sklearn.preprocessing as sp


def get_data_set():
    # set percentage of training data here
    percentage_train = 60
    percentage_test = 100 - percentage_train

    # UNCOMMENT THIS SECTION TO CLUSTER TO TWO GROUPS
    # load LOS and NLOS channel arrays
    LOS_1 = np.load('LOS_VAE.npy')
    LOS_2 = np.load('LOS_2_vae.npy')
    NLOS_1 = np.load('NLOS_VAE.npy')
    NLOS_2 = np.load('NLOS_2_vae.npy')

    # number of vectors in each data set
    n_LOS_1 = np.size(LOS_1, axis=0)
    n_LOS_2 = np.size(LOS_2, axis=0)
    n_NLOS_1 = np.size(NLOS_1, axis=0)
    n_NLOS_2 = np.size(NLOS_2, axis=0)

    # separating train and test sets
    n_train_LOS_1 = int(np.floor(n_LOS_1 * (percentage_train / 100)))
    n_train_LOS_2 = int(np.floor(n_LOS_2 * (percentage_train / 100)))
    n_train_NLOS_1 = int(np.floor(n_NLOS_1 * (percentage_train / 100)))
    n_train_NLOS_2 = int(np.floor(n_NLOS_2 * (percentage_train / 100)))

    n_test_LOS_1 = n_LOS_1 - n_train_LOS_1
    n_test_LOS_2 = n_LOS_2 - n_train_LOS_2
    n_test_NLOS_1 = n_NLOS_1 - n_train_NLOS_1
    n_test_NLOS_2 = n_NLOS_2 - n_train_NLOS_2

    X_train = np.concatenate((LOS_1[0:n_train_LOS_1, :], LOS_2[0:n_train_LOS_2, :],
                              NLOS_1[0:n_train_NLOS_1, :], NLOS_2[0:n_train_NLOS_2, :]), axis=0)
    X_test = np.concatenate((LOS_1[n_LOS_1 - n_test_LOS_1:n_LOS_1, :], LOS_2[n_LOS_2 - n_test_LOS_2:n_LOS_2, :],
                             NLOS_1[n_NLOS_1 - n_test_NLOS_1:n_NLOS_1, :], NLOS_2[n_NLOS_2 - n_test_NLOS_2:n_NLOS_2, :]
                             ), axis=0)

    # create labels, 0,1 for LOS and 2,3 for NLOS
    y_train = np.concatenate((np.zeros((n_train_LOS_1, 1), dtype=int),
                              np.ones((n_train_LOS_2, 1), dtype=int),
                              2 * np.ones((n_train_NLOS_1, 1), dtype=int),
                              3 * np.ones((n_train_NLOS_2, 1), dtype=int)), axis=0)
    y_test = np.concatenate((np.zeros((n_test_LOS_1, 1), dtype=int),
                             np.ones((n_test_LOS_2, 1), dtype=int),
                             2 * np.ones((n_test_NLOS_1, 1), dtype=int),
                             3 * np.ones((n_test_NLOS_2, 1), dtype=int)), axis=0)

    scaler = sp.MinMaxScaler(feature_range=[-1, 1])
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.fit_transform(X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test

    del LOS_1, LOS_2, NLOS_1, NLOS_2, X_train, X_test

    # create training tensors
    X_train_scaled = torch.stack([torch.from_numpy(np.array(i)) for i in X_train_scaled])
    y_train = torch.stack([torch.from_numpy(np.array(i)) for i in y_train])

    # create testing tensors
    X_test_scaled = torch.stack([torch.from_numpy(np.array(i)) for i in X_test_scaled])
    y_test = torch.stack([torch.from_numpy(np.array(i)) for i in y_test])

    train_data_set = torch.utils.data.TensorDataset(X_train_scaled, y_train)
    test_data_set = torch.utils.data.TensorDataset(X_test_scaled, y_test)

    # create array that stores information of number of total, training and testing channels
    param_data_set = np.array([n_LOS_1, n_LOS_2, n_NLOS_1, n_NLOS_2,
                               n_train_LOS_1, n_train_LOS_2, n_train_NLOS_1, n_train_NLOS_2,
                               n_test_LOS_1, n_test_LOS_2, n_test_NLOS_1, n_test_NLOS_2])
    del n_LOS_1, n_LOS_2, n_NLOS_1, n_NLOS_2, n_train_LOS_1, n_train_LOS_2, n_train_NLOS_1, n_train_NLOS_2, \
        n_test_LOS_1, n_test_LOS_2, n_test_NLOS_1, n_test_NLOS_2

    return train_data_set, test_data_set, param_data_set

# if __name__ == '__main__':
#     channel_LOS = np.load('LOS_VAE.npy')
#     channel_NLOS = np.load('NLOS_VAE.npy')
#     print(np.shape(channel_LOS))
#     print(np.shape(channel_NLOS))
