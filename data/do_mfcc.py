import librosa
import os
import numpy as np
from scipy.io import loadmat, savemat


def do_mfcc_d_d2(X, sr=44100):
    res = []
    for i in X:
        y = (i[:sr] + i[sr:]) / 2
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        delta1 = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        res.append(np.concatenate([mfccs, delta1, delta2], axis=0))
    return np.array(res, dtype='float32')


def apply_to_files(inpath=os.curdir):
    for i in os.listdir(inpath):
        data = loadmat(os.path.join(inpath, i))
        X = do_mfcc_d_d2(data['X'])
        data['X'] = X
        savemat(i, data)


def main():
    apply_to_files(os.curdir)


if __name__ == '__main__':
    main()
