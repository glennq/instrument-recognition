import numpy as np
import pandas as pd
import cPickle


def reverse_annotation(annotations):
    res = {}
    for k, v in annotations.items():
        for j in v.columns[1:]:
            if j not in res:
                res[j] = set([k])
            else:
                res[j].add(k)
    return res


def read_annotations(fpath):
    with open(fpath) as f:
        annotations = cPickle.load(f)
    return annotations


if __name__ == '__main__':
    """Assumes that anno_label.pkl is already produced by data_prep.py
    """
    annotations = read_annotations('anno_label.pkl')
    rannotation = reverse_annotation(annotations)
    keys = rannotation.keys()
    keys.sort(key=lambda x: len(rannotation[x]))
    for k in keys:
        print '{}:'.format(k)
        for i in rannotation[k]:
            print '    {}'.format(i)
