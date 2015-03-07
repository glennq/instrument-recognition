import scipy.io
import numpy as np
from pylearn2.datasets import dense_design_matrix
from theano import config

"""
Scripts to assist reading data in pylearn2.
"""


class MedleyDB(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path, which_set):
        data = scipy.io.loadmat(path)
        axes = ('b', 0, 1, 'c')
        view_converter = dense_design_matrix.DefaultViewConverter((1, 44100,
                                                                   2),
                                                                  axes)
        super(MedleyDB, self).__init__(X=np.cast[config.floatX](data[which_set
                                                                     + '_X']),
                                       y=np.cast[config.floatX](data[which_set
                                                                     + '_y']),
                                       view_converter=view_converter)


def train_test_split(path):
    data = scipy.io.loadmat(path)
    N = len(data['y'])
    index = range(N)
    np.random.shuffle(index)
    train_X = data['X'][index[:int(N*0.8)]]
    train_y = data['y'][index[:int(N*0.8)]]
    test_X = data['X'][index[int(N*0.8):]]
    test_y = data['y'][index[int(N*0.8):]]
    del data
    data = {'train_X': train_X, 'train_y': train_y,
            'test_X': test_X, 'test_y': test_y}
    scipy.io.savemat(path + '_splitted', data)
