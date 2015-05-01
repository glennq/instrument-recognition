import numpy as np
import pandas as pd
import cPickle
import sys


def reverse_annotation(annotations, grouping):
    res = {}
    for k, v in annotations.items():
        for j in v:
            instr = grouping[j] if j in grouping else j
            if instr not in res:
                res[instr] = set([k])
            else:
                res[instr].add(k)
    return res


def read_annotations(fpath):
    with open(fpath) as f:
        annotations = cPickle.load(f)
    return annotations


def read_groupings(fpath, num):
    groups = pd.read_csv(fpath, index_col=0)
    grouping = dict(zip(groups['Instrument'].values,
                        groups['Group {}'.format(num)].values))
    return grouping


def num_nonrare_inst(rannotation, thres):
    all_songs = set()
    songs = set()
    cnt = 0
    for k, v in rannotation.items():
        all_songs.update(v)
        if len(v) < thres:
            songs.update(v)
        else:
            cnt += 1
    return cnt, len(all_songs.difference(songs))


if __name__ == '__main__':
    """Assumes that anno_label.pkl is already produced by data_prep.py
    """
    annotations = read_annotations('annotation_investigator/song_instr.pkl')
    grouping = read_groupings('instGroup.csv', int(sys.argv[1]))
    rannotation = reverse_annotation(annotations, grouping)
    keys = rannotation.keys()
    keys.sort(key=lambda x: len(rannotation[x]))
    for k in keys:
        print '{}:'.format(k)
        for i in rannotation[k]:
            print '    {}'.format(i)
    n_nonrare_inst, n_songs_nonrare_inst = num_nonrare_inst(rannotation,
                                                            int(sys.argv[2]))
    print 'Number of songs with no instruments appearing in less than {} songs = {}'. \
        format(sys.argv[2], n_songs_nonrare_inst)
    print 'Number of instrument appearing in at least {} songs = {}'. \
        format(sys.argv[2], n_nonrare_inst)
