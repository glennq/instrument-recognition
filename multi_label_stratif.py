import numpy as np


"""
This file contains the method to randomly split the set of all songs into
training and test set with stratification.

Based on the paper titled "On the Stratification of Multi-Label Data"
(Sechidis et al., 2011)
which is accessible from the following URL:
http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

WARNING: Code not tested
"""


def multi_label_stratif(self, labels, num_split=2, p_split=[0.8, 0.2],
                        rand_state=None):
    """Split the data according to labels
    Args:
        labels (list of lists): each inside list contains the labels for each
                                sample in dataset
        num_split (int): the number of subsets to split into
        p_split (list of floats): each element represents the proportion of
                                  dataset to be splitted into corresponding
                                  subset. Must sum to one and length==num_split
        rand_state (int): the random state to use
    Returns:
        splits (list of lists): length=num_split, each containing the indices
                                of samples splitted to this particular subset
    """
    # check inputs
    assert(len(p_split) == num_split)
    assert(sum(p_split) == 1)

    # set random state
    rng = np.random.RandomState(rand_state)

    # initialize
    N = len(labels)
    L = set()  # set of all possible labels
    n_label = {}  # examples containing each label
    for i, e in enumerate(labels):
        L.update(i)
        for j in e:
            if j not in n_label:
                n_label[j] = set()
            else:
                n_label[j].add(i)
    index = set(range(N))
    splits = []
    for i in range(num_split):
        splits.append([])

    # desired number of examples for each subset
    c_subset = []
    for i in range(num_split):
        c_subset.append(N * p_split[i])

    # desired number of examples of each label at each subset
    c_label_subset = {}
    for e in L:
        c_label_subset[e] = []
        for j in range(num_split):
            c_label_subset[e].append(len(n_label[e]) * p_split[j])

    while (len(index) > 0):
        # Find the label with the fewest (but at least one) remaining examples
        min_l = [labels[0][0]]
        for e in L:
            if len(n_label[e]) == 0:
                continue
            else:
                if len(n_label[e]) < len(n_label[min_l[0]]) or \
                   len(n_label[min_l[0]]) == 0:
                    min_l = [e]
                elif len(n_label[e]) == len(n_label[min_l[0]]) and e not in min_l:
                    min_l.append(e)
        # randomly break tie
        l = min_l[rng.randint(len(min_l))]

        for i in n_label[l]:
            # Find the subset(s) with the largest number of desired examples
            # for this label
            M = [0]
            for j in range(1, num_split):
                if c_label_subset[l][j] > c_label_subset[l][M[0]]:
                    M = [j]
                elif c_label_subset[l][j] == c_label_subset[l][M[0]]:
                    M.append(j)
            if len(M) == 1:
                m = M[0]
            else:
                # breaking ties by considering the largest number of
                # desired examples
                M_prime = [M[0]]
                for j in M[1:]:
                    if c_subset[j] > c_subset[M_prime[0]]:
                        M_prime = [j]
                    elif c_subset[j] > c_subset[M_prime[0]]:
                        M_prime.append(j)
                if len(M_prime) == 1:
                    m = M_prime[0]
                else:
                    # break further tie randomly
                    m = M_prime[rng.randint(len(M_prime))]

            # assign and update
            splits[m].append(i)
            assert(i in index)
            index.discard(i)
            for e in labels[i]:
                n_label[e].discard(i)
                c_label_subset[e][m] -= 1
            c_subset[m] -= 1

    return splits
