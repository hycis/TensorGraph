
import numpy as np
import os, pickle


def _compute_zca_transform(imgs, filter_bias=0.1):
    """
    Compute the zca whitening transform matrix.
    """
    print("Computing ZCA transform matrix")
    meanX = np.mean(imgs, 0)

    covX = np.cov(imgs.T)
    D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0

    D = D ** -.5

    W = np.dot(E, np.dot(np.diag(D), E.T))
    return meanX, W


def zca_whiten(train, test, cache=None):
    """
    Use train set statistics to apply the ZCA whitening transform to
    both train and test sets.
    """
    if cache and os.path.isfile(cache):
        with open(cache, 'rb') as f:
            (meanX, W) = pickle.load(f)
    else:
        meanX, W = _compute_zca_transform(train)
        if cache:
            print("Caching ZCA transform matrix")
            with open(cache, 'wb') as f:
                pickle.dump((meanX, W), f, 2)

    print("Applying ZCA whitening transform")
    train_w = np.dot(train - meanX, W)
    test_w = np.dot(test - meanX, W)

    return train_w, test_w


def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
    """
    Subtract mean and normalize by vector norm.
    """

    X = X - X.mean(axis=1)[:, np.newaxis]

    normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]

    return X
