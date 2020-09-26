import pickle
from functools import partialmethod
from pathlib import Path

import numpy as np
import tables


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
    def __init__(
        self,
        root_path,
        fold_index,
        subject_ids: list,
        h5_datafile_path: Path,
        npz_mips_file_path: Path
    ):
        self.root_path = root_path
        self.fold_index = fold_index
        self.subject_ids = subject_ids
        self.h5_datafile_path = h5_datafile_path
        self.npz_mips_file_path = npz_mips_file_path

    def load_indices_for_category(self, category):
        path = Path(self.root_path, "fold_{}_{}.pkl".format(self.fold_index, category))
        with open(str(path), "rb") as f:
            h5_indices = pickle.load(f)

        desired_subject_ids = self._load_subject_ids_from_h5_datafile(h5_indices)
        npz_indices_by_subject_id = self._index_load_npz_indices_by_subject_id()

        return self._find_npz_indices_for_desired_subject_ids(
            npz_indices_by_subject_id,
            desired_subject_ids
        )


    def _find_npz_indices_for_desired_subject_ids(self,
                                                  npz_indices_by_subject_id,
                                                  subject_ids):
        result = []
        for subject_id in subject_ids:
            result.append(npz_indices_by_subject_id[subject_id])
        return result

    def _index_load_npz_indices_by_subject_id(self):
        npz_indices_by_subject_id = {}
        with np.load(str(self.npz_mips_file_path)) as data:
            for index, subject_id in enumerate(data['names']):
                npz_indices_by_subject_id[subject_id] = index
        return npz_indices_by_subject_id

    def _load_subject_ids_from_h5_datafile(self, h5_indices):
        subject_ids = []
        with tables.open_file(str(self.h5_datafile_path)) as df:
            for index in h5_indices:
                subject_ids.append(df.root.subject_ids[index].decode('utf-8'))
        return subject_ids

    # L3 Finder doesn't use validation within training
    # def get_train_indices(self):
    #     train = self.load_indices_for_category("train")
    #     val = self.load_indices_for_category("val")
    #     return np.concatenate(
    #         [
    #             self.load_indices_for_category("train"),
    #             self.load_indices_for_category("val"),
    #         ]
    #     )
    get_train_indices = partialmethod(load_indices_for_category, "train")
    get_val_indices = partialmethod(load_indices_for_category, "val")
    get_test_indices = partialmethod(load_indices_for_category, "test")
