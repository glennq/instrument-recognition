import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.io import loadmat


def do_train(X, y):
    clf = OneVsRestClassifier(SVC(kernel='linear'), -1)
    clf.fit(X, y)
    return clf


def do_test(clf, X, y, p, savename):
    pred = clf.predict(X)
    accu = (pred == y).mean()
    exact_accu = sum([((y[i] == e).sum() == y.shape[1])
                      for i, e in enumerate(pred)]) / float(len(y))
    mod_accu = ((pred == y) * p).mean()
    all_precision = (pred * y).sum() / pred.sum()
    all_recall = (pred * y).sum() / y.sum()
    f_micro = 2 * (pred * y).sum() / (pred.sum() + y.sum())
    f_macro = (2 * (y * pred).sum(0) / (y.sum(0) + pred.sum(0))).mean()
    res = '''
    Accuracy = {}\nExact match Accuracy = {}\nModified accuracy = {}\n
    Precision = {}\nRecall = {}\nF micro = {}\nF macro = {}
    '''.format(accu, exact_accu, mod_accu, all_precision, all_recall,
               f_micro, f_macro)
    print res
    with open(savename, 'wb') as f:
        f.write(res)


def main():
    train = loadmat('/scratch/jq401/ml-mat-split-3943485/train.mat')
    test = loadmat('/scratch/jq401/ml-mat-split-3943485/test.mat')
    train['y'] = np.float32(train['y'] > 0.5)
    test['y'] = np.float32(test['y'] > 0.5)
    clf = do_train(train['X'], train['y'])
    do_test(clf, train['X'], train['y'], train['p'], 'train.log')
    do_test(clf, test['X'], test['y'], test['p'], 'test.log')


if __name__ == '__main__':
    main()
