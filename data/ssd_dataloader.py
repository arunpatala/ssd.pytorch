"""SSD Dataloader factory

Returns the corresponding dataloader object for use in the SSD network

Ellis Brown
"""

import torch.utils.data as data
from . import SHUFFLE, WORKERS, BATCHES
from .datasets import detection_collate


def ssd_dataloader(dataset, batch_size=BATCHES, num_workers=WORKERS):
    """SSD Dataloader factory

    Creates a dataloader that will work with SSD detection

    Arguments:
        dataset (Dataset): dataset to use, should be created via ssd_dataset()
        batch_size (int, optional): number of elements in each batch
            (default: BATCHES from config file)
        num_workers (int, optional): how many subprocesses to use for data
            loading (default: BATCHES from config file)
    Return:
        the specified dataset object for use in the SSD network
    """

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=SHUFFLE,
                           num_workers=num_workers, collate_fn=detection_collate)
