import random
import logging
import json

from typing import Optional, Tuple, Callable, Dict, List, Union

import numpy as np
import glob

from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable

from util import TwoCropTransform
from datasets.imagefolder import ImageFolderWithPath
from datasets.preprocess.transform import Compose

# Initiate Logger
logger = logging.getLogger(__name__)


class DERMADataset(Dataset):
    def __init__(self,
                data_location: str,
                transform: Optional[Tuple[Callable, ...]] = None):
        
        # Save data location
        self.data_location = data_location

        # Preprocess functions
        self.transform = transform

        self.data = ImageFolderWithPath(self.data_location)
        self.data_cnt = len(self.data)
        self.class_to_idx = self.data.class_to_idx.items()
        self.target = np.array([record[1] for record in self.data])
        self.data_classes = np.unique(self.target)
        self.class_cnt = len(self.data_classes)
        self.cid2name = {v: k for k, v in self.data.class_to_idx.items()}
        self.name2cid = self.data.class_to_idx

    def __getitem__(self, idx):
        img, category, path = self.data[idx]
        if self.transform is not None:
            # img = Compose(self.transform)(img)
            img = TwoCropTransform(self.transform)(img)
        return img, category, path

    def __len__(self):
        return len(self.data)


    def _get_mean_std(self, train_valid_id) -> Dict[str, float]:
        ## Calculate mean and std in the training process
        ## It leads to slow computing
        # mean, std = 0., 0.
        # for idx in train_valid_id:
        #     img = transforms.Resize([112, 112])(self.data[idx][0])
        #     img = transforms.ToTensor()(img)
        #     img = img.view(img.size(0), -1)
        #     mean += img.mean(1)
        #     std += img.std(1)
        # mean /= len(train_valid_id)
        # std /= len(train_valid_id)
        # return {
        #     'mean': mean,
        #     'std': std
        # }

        return {
            'mean': np.array([0.6678, 0.5300, 0.5245]),
            'std': np.array([0.1320, 0.1462, 0.1573])
        }

    def __getitem__(self, idx):
        data, category = self.data[idx][0], self.data[idx][1]
        if self.transform is not None:
            data = Compose(self.transform)(data)
        return data, category

    def __len__(self):
        return self.data_cnt

