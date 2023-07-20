import random
import numpy as np
import logging

## import cupy as cp
## from cupyx.scipy import ndimage
from scipy import ndimage

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PeakDataset(Dataset):

    def __init__(self, data_list, size_sample, trans_list = None, reverse_bg = False, mpi_comm = None, returns_mask = False):
        self.data_list    = data_list
        self.size_sample  = size_sample
        self.trans_list   = trans_list
        self.reverse_bg   = reverse_bg
        self.mpi_comm     = mpi_comm
        self.returns_mask = returns_mask


        self._structure = np.array([[1,1,1],
                                    [1,1,1],
                                    [1,1,1]])

        self.idx_sample_list = self.build_dataset()
        self.dataset_cache_dict = {}

        return None


    def __len__(self):
        return self.size_sample


    def build_dataset(self):
        data_list     = self.data_list    # N, B, C, H, W
        size_sample   = self.size_sample

        size_data_list  = len(data_list)
        candidate_list  = range(size_data_list)
        idx_sample_list = random.choices(candidate_list, k = size_sample)

        return idx_sample_list


    def get_data(self, idx):
        structure = self._structure

        trans_list = self.trans_list
        idx_sample = self.idx_sample_list[idx]

        data = self.data_list[idx_sample]
        batch_data = np.concatenate(data, axis = 0)

        if trans_list is not None:
            for trans in trans_list:
                batch_data = trans(batch_data)

        img   = batch_data[0:1]
        label = batch_data[1: ]

        _, num_peaks = ndimage.label(label[0]==1.0, structure = structure)

        if self.reverse_bg:
            label[-1] = 1 - label[-1]

        # Normalize input image...
        img_mean = np.nanmean(img)
        img_std  = np.nanstd(img)
        img      = img - img_mean

        if img_std == 0:
            img_std  = 1.0
            label[:] = 0

        img /= img_std

        return img, label, num_peaks


    def __getitem__(self, idx):
        returns_mask = self.returns_mask

        img, label, num_peaks = self.dataset_cache_dict[idx]      \
                                if idx in self.dataset_cache_dict \
                                else self.get_data(idx)

        ret = (img, label, np.array([num_peaks])) if returns_mask else (img, np.array([num_peaks]))

        return ret




class HitDataset(Dataset):

    def __init__(self, data_list, size_sample, min_num_peaks = 15, trans_list = None, reverse_bg = False, mpi_comm = None, returns_mask = False):
        self.data_list     = data_list
        self.size_sample   = size_sample
        self.trans_list    = trans_list
        self.reverse_bg    = reverse_bg
        self.mpi_comm      = mpi_comm
        self.returns_mask  = returns_mask
        self.min_num_peaks = min_num_peaks


        self._structure = np.array([[1,1,1],
                                    [1,1,1],
                                    [1,1,1]])

        self.idx_sample_list = self.build_dataset()
        self.dataset_cache_dict = {}

        return None


    def __len__(self):
        return self.size_sample


    def build_dataset(self):
        data_list     = self.data_list    # N, B, C, H, W
        size_sample   = self.size_sample

        size_data_list  = len(data_list)
        candidate_list  = range(size_data_list)
        idx_sample_list = random.choices(candidate_list, k = size_sample)

        return idx_sample_list


    def get_data(self, idx):
        min_num_peaks = self.min_num_peaks
        structure = self._structure

        trans_list = self.trans_list
        idx_sample = self.idx_sample_list[idx]

        data = self.data_list[idx_sample]
        batch_data = np.concatenate(data, axis = 0)

        if trans_list is not None:
            for trans in trans_list:
                batch_data = trans(batch_data)

        img   = batch_data[0:1]
        label = batch_data[1: ]

        _, num_peaks = ndimage.label(label[0]==1.0, structure = structure)

        if self.reverse_bg:
            label[-1] = 1 - label[-1]

        # Normalize input image...
        img_mean = np.nanmean(img)
        img_std  = np.nanstd(img)
        img      = img - img_mean

        if img_std == 0:
            img_std  = 1.0
            label[:] = 0

        img /= img_std

        is_hit = num_peaks > min_num_peaks

        return img, label, is_hit


    def __getitem__(self, idx):
        returns_mask = self.returns_mask

        img, label, is_hit = self.dataset_cache_dict[idx]      \
                             if idx in self.dataset_cache_dict \
                             else self.get_data(idx)

        ret = (img, label, np.array([is_hit], dtype = int)) if returns_mask else (img, np.array([is_hit], dtype = int))

        return ret
