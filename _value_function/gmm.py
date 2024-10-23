import pathlib
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_prior():

    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    fpath = pathlib.Path(f'{CCAI_PATH}/data')

    filename = '/initial_poses/initial_poses_10k.pkl'
    
    with open(f'{fpath.resolve()}/{filename}', 'rb') as file:
        poses  = pkl.load(file)
    
    inputs = np.array([t.numpy() for t in poses]).reshape(-1, 20)
    print(inputs.shape)

    return inputs

if __name__ == "__main__":

    data = get_prior()

    # mean = np.mean(data, axis=0)


    # Assume X is your data (N samples, D dimensions)
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(data)

    # Extract the parameters
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Plot the data points and the GMM means
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=10, label='Data')
    plt.scatter(pca.transform(gmm.means_)[:, 0], pca.transform(gmm.means_)[:, 1], c='red', marker='x', s=100, label='GMM Means')
    plt.title('2D Visualization of GMM')
    plt.legend()
    plt.show()