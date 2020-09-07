import pickle
from functools import partialmethod
from pathlib import Path

import numpy as np


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class Fold:
    def __init__(self, root_path, fold_index, subject_ids: list):
        self.root_path = root_path
        self.fold_index = fold_index
        self.subject_ids = subject_ids

    def load_indices_for_category(self, category):
        path = Path(self.root_path, "fold_{}_{}.pkl".format(self.fold_index, category))
        with open(path, "rb") as f:
            return pickle.load(f)
        # indices = np.array([self.subject_ids.index(str(i)) for i in fold_subject_ids])
        # return indices

    get_train_indices = partialmethod(load_indices_for_category, "train")
    get_val_indices = partialmethod(load_indices_for_category, "val")
    get_test_indices = partialmethod(load_indices_for_category, "test")
