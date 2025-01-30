from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from typing import Optional
from PIL import Image

import torch
import numpy as np


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, imagenet_preprocess=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.imagenet_preprocess = imagenet_preprocess

    def __call__(self, gt_im):
        if self.imagenet_preprocess:
            pil_image = Image.fromarray(np.uint8(np.transpose(gt_im.numpy(), (1, 2, 0)) * 255))
            image_size = 256

            while min(*pil_image.size) >= 2 * image_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image)
            crop_y = (arr.shape[0] - image_size) // 2
            crop_x = (arr.shape[1] - image_size) // 2
            gt_im = np.transpose(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size, :], (2, 0, 1))

            gt = torch.tensor(gt_im / 127.5 - 1)
        else:
            gt = 2 * gt_im - 1 # already in [0,1]

        return gt.float()


class ImageDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_config):
        super().__init__()
        self.prepare_data_per_node = True
        self.data_config = data_config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        transform = transforms.Compose([transforms.ToTensor(), DataTransform(self.data_config["preprocess"])])

        # Split into 1k val set for lr tune
        full_data = datasets.ImageFolder(self.data_config["full_data_path"], transform=transform)
        test_data = torch.utils.data.Subset(full_data, range(self.data_config["test_start"], self.data_config["test_end"]))
        lr_tune_data = torch.utils.data.Subset(full_data, range(self.data_config["tune_start"], self.data_config["tune_end"]))

        self.full_data, self.lr_tune_data, self.test_data = full_data, lr_tune_data, test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.full_data,
            batch_size=self.data_config["batch_size"],
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.lr_tune_data,
            batch_size=self.data_config["batch_size"],
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.data_config["batch_size"],
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
