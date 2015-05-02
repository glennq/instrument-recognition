import os
import numpy as np
from scipy.io import loadmat, savemat


def fit_gaussian(X):
    """Use MLE to fit gaussian parameters
    Args:
        X (np.array): each column represents a sample point
    Returns:
        myu (np.array): estimated guassian mean
        sigma (np.array): esitmated gaussian covariance matrix
    """
    myu = X.mean(axis=1)
    centered = X - myu.reshape(len(myu), 1)
    sigma = centered.dot(centered.T) / (X.shape[1] - 1)
    return myu, sigma


def gaussian_transform(fname):
    f = loadmat(fname)
    res = []
    for i in f['X']:
        a, b = fit_gaussian(i)
        res.append(np.hstack([a.ravel(), b.ravel()]))
    f['X'] = np.array(res)
    savemat(fname, f)


def main():
    gaussian_transform('/scratch/jq401/ml-mat-split-3943485/train.mat')
    gaussian_transform('/scratch/jq401/ml-mat-split-3943485/test.mat')


if __name__ == '__main__':
    main()
