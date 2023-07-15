import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from config import Config
from torch.utils.data import Dataset, DataLoader
args = Config()

class Train(Dataset):
    def __init__(self, csv_path, aug_save_path):
        """
        Constructs train dataset.
        @param csv_path: path to sampled dataset csv
        @param data_type: type of data which is about to be trained - histograms or images
        @param num_augs: number of augmentation functions to be sampled
        @param save_augs: flag to save or not save the augmented data points.
        """
        super().__init__()
        # Read the train csv file
        self.data_info = pd.read_csv(csv_path)

        # Read csv columns into self variables
        self.img_paths = np.asarray(self.data_info['file_path'])
        self.labels = np.asarray(self.data_info['label'])

        # global mean - 111, global std = 27
        self.aug_save_path = aug_save_path

        # Calculate length of data set
        self.data_len = len(self.img_paths)

        # Define tensor conversion and normalization for images in __getitem__, if you wish to use these augmentations in the train data, remove hashtags
        self.transform = transforms.Compose([
            # Augment image
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomVerticalFlip(p=0.3),
            # transforms.RandomRotation(degrees=45),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[111], std=[27]),
        ])

    def __getitem__(self, index):
        """
        this method is approached to when each data point is loaded through the batch creation process.
        @param index: index in the sampled dataset
        @return: augmented data, label and data path and study id.
        """
        single_data_path = self.img_paths[index]
        single_data_label = self.labels[index]

        # Load image
        gray_img = cv2.cvtColor(cv2.imread(single_data_path), cv2.COLOR_BGR2RGB)

        # Assert label is 'sick' or 'healthy'
        if single_data_label == "init":
            single_data_label = 0
        elif single_data_label == "final":
            single_data_label = 1

        assert single_data_label == 1 or single_data_label == 0, f'{single_data_path}: label is not 1 or 0.'

        # Transform image into a normalized tensor
        single_data_tensor = self.transform(gray_img)
        return single_data_tensor, single_data_label

    def __len__(self):
        """
        This method is approached to when the length of the dataset is called.
        @return: length of the dataset
        """
        return self.data_len

class Validation(Dataset):
    """
    This is the train dataset which load the data tha has been smapled already.
    """
    def __init__(self, csv_path):
        """
        Constructs train dataset.
        @param csv_path: path to sampled dataset csv
        @param data_type: type of data which is about to be trained - histograms or images
        @param num_augs: number of augmentation functions to be sampled
        @param save_augs: flag to save or not save the augmented data points.
        """
        super().__init__()
        # Read the train csv file
        self.data_info = pd.read_csv(csv_path)

        # Read csv columns into self variables
        self.img_paths = np.asarray(self.data_info['file_path'])
        self.labels = np.asarray(self.data_info['label'])

        # global mean - 111, global std = 27
        # Calculate length of data set
        self.data_len = len(self.img_paths)

        # Define tensor conversion and normalization for images in __getitem__
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[111], std=[27]),
        ])

    def __getitem__(self, index):
        """
        this method is approached to when each data point is loaded through the batch creation process.
        @param index: index in the sampled dataset
        @return: augmented data, label and data path and study id.
        """
        single_data_path = self.img_paths[index]
        single_data_label = self.labels[index]

        # Load image
        gray_img = cv2.cvtColor(cv2.imread(single_data_path), cv2.COLOR_BGR2RGB)

        if single_data_label == "init":
            single_data_label = 0
        elif single_data_label == "final":
            single_data_label = 1

        # Assert label is 'sick' or 'healthy'
        assert single_data_label == 1 or single_data_label == 0, f'{single_data_path}: label is not 1 or 0.'

        # Transform image into a normalized tensor
        single_data_tensor = self.transform(gray_img)

        return single_data_tensor, single_data_label

    def __len__(self):
        """
        This method is approached to when the length of the dataset is called.
        @return: length of the dataset
        """
        return self.data_len


def get_balanced_dataloader(dataset, batch_size, num_workers):
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      sampler=torch.utils.data.sampler.WeightedRandomSampler(
          dataset.weights, len(dataset)),
      num_workers=num_workers)
  return dataloader


