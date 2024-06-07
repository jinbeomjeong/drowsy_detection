import os, torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor


class AiHubEyesTestDataset(Dataset):
    def __init__(self, img_root_path: str, driver_name: str, transform: transforms):
        self.__transform = transform
        self.__target_dataset = list()
        eye_img_dataset_list = list()
        target_driver_name = driver_name

        for (path_1, dirs_1, files_1) in os.walk(img_root_path):
            for sub_dir in dirs_1:
                eye_label = 1 if sub_dir == 'close' else 0

                for (path_2, dirs_2, files_2) in os.walk(img_root_path + os.sep + sub_dir):
                    for file_name in files_2:
                        eye_img_dataset_list.append([img_root_path + os.sep + sub_dir + os.sep + file_name, eye_label])

        if target_driver_name:
            for img_path, label in eye_img_dataset_list:
                driver_name = os.path.splitext(os.path.basename(img_path))[0]

                for i in range(10):
                    idx = driver_name.find('_')
                    driver_name = driver_name[idx + 1:]

                if driver_name == target_driver_name:
                    self.__target_dataset.append([img_path, label])
        else:
            self.__target_dataset = eye_img_dataset_list

    def __len__(self):
        return len(self.__target_dataset)

    def __getitem__(self, idx):
        eye_img = self.__transform(Image.open(self.__target_dataset[idx][0]))

        return eye_img, torch.FloatTensor(self.__target_dataset[idx][1])


class AiHubEyeDataModule(pl.LightningDataModule):
    def __init__(self, train_eye_label_desc_path: str, val_eye_label_desc_path: str, eye_img_root_path: str, target_driver_name: str,
                 execute_prepare_data: bool = False, batch_size: int = 16, n_workers: int = 8):
        super().__init__()
        self.__h_offset_scale: float = 0.08
        self.__v_offset_scale: float = 0.08
        self.__eye_img_files = list()
        self.__left_eye_pos = list()
        self.__right_eye_pos = list()
        self.__left_eye_label_list = list()
        self.__right_eye_label_list = list()

        self.__train_eye_label_desc_path = train_eye_label_desc_path
        self.__val_eye_label_desc_path = val_eye_label_desc_path
        self.__eye_img_root_path = eye_img_root_path
        self.__execute_prepare_data = execute_prepare_data
        self.__target_driver_name = target_driver_name

        self.__train_dataset = list()
        self.__val_dataset = list()

        self.batch_size = batch_size
        self.num_workers = n_workers
        self.__transform = transforms.Compose([transforms.Resize((64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    def prepare_data(self):
        if self.__execute_prepare_data:
            train_eye_label_df = pd.read_csv(self.__train_eye_label_desc_path)
            val_eye_label_df, test_eye_label_df = train_test_split(pd.read_csv(self.__val_eye_label_desc_path), test_size=0.5, random_state=0)

            for i in tqdm(range(train_eye_label_df.shape[0]), desc='preparing train dataset'):
                file_path = train_eye_label_df.loc[i, 'file_name']
                eye_pos = train_eye_label_df.loc[i, ['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
                eye_label = train_eye_label_df.loc[i, 'closed']

                img = Image.open(file_path)
                eye_img = img.crop(tuple(eye_pos))
                eye_img.save(self.__eye_img_root_path + os.path.sep + 'train' + os.path.sep + str(eye_label) +
                             os.path.sep + os.path.splitext(os.path.basename(file_path))[0] + '_' + str(i) + '_' + '.jpg', format='JPEG')

            for i in tqdm(range(val_eye_label_df.shape[0]), desc='preparing val dataset'):
                file_path = val_eye_label_df.loc[i, 'file_name']
                eye_pos = val_eye_label_df.loc[i, ['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
                eye_label = val_eye_label_df.loc[i, 'closed']

                img = Image.open(file_path)
                eye_img = img.crop(tuple(eye_pos))
                eye_img.save(self.__eye_img_root_path + os.path.sep + 'val' + os.path.sep +
                             str(eye_label) + os.path.sep + os.path.splitext(os.path.basename(file_path))[0] + '_' + str(i) + '_' + '.jpg', format='JPEG')

            for i in tqdm(range(test_eye_label_df.shape[0]), desc='preparing test dataset'):
                file_path = test_eye_label_df.loc[i, 'file_name']
                eye_pos = test_eye_label_df.loc[i, ['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
                eye_label = test_eye_label_df.loc[i, 'closed']
                driver_type = file_path.split(os.sep)[-3]

                img = Image.open(file_path)
                eye_img = img.crop(tuple(eye_pos))
                eye_img.save(self.__eye_img_root_path + os.path.sep + 'test' + os.path.sep +
                             str(eye_label) + os.path.sep + os.path.splitext(os.path.basename(file_path))[0] + '_' + str(i) + '_' + driver_type + '.jpg', format='JPEG')

    def setup(self, stage: str):
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.RandomAffine(degrees=(-5, 5)),
                                        transforms.RandomPerspective(0.5),
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

        self.__train_dataset = ImageFolder(root=self.__eye_img_root_path + os.sep + 'train', transform=transform)
        self.__val_dataset = ImageFolder(root=self.__eye_img_root_path + os.sep + 'val', transform=self.__transform)
        self.__test_dataset = AiHubEyesTestDataset(img_root_path=self.__eye_img_root_path + os.sep + 'test',
                                                   driver_name=self.__target_driver_name, transform=self.__transform)

    def train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.__val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.__test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def __save_img(self, eye_label_df: pd.Series):
        file_path = eye_label_df['file_name']
        eye_pos = eye_label_df[['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
        eye_label = eye_label_df['closed']

        img = Image.open(file_path)
        eye_img = img.crop(tuple(eye_pos))
        eye_img.save(self.__eye_img_root_path + os.path.sep + 'train' + os.path.sep + str(eye_label) +
                     os.path.sep + os.path.splitext(os.path.basename(file_path))[0] + '_' + '_' + '.jpg', format='JPEG')

    def save_img_executor(self, eye_label_df: pd.DataFrame, n_of_workers: int, desc: str):
        with ThreadPoolExecutor(max_workers=n_of_workers) as executor:
            ds_list = [row for i, row in eye_label_df.iterrows()]
            tqdm(executor.map(self.__save_img, ds_list), desc=desc, total=eye_label_df.shape[0])

    def calculate_class_weight(self):
        train_dataset = ImageFolder(root=self.__eye_img_root_path + os.sep + 'train', transform=self.__transform)
        val_dataset = ImageFolder(root=self.__eye_img_root_path + os.sep + 'val', transform=self.__transform)
        test_dataset = ImageFolder(root=self.__eye_img_root_path + os.sep + 'test', transform=self.__transform)
        full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

        label_list = []

        for img, label in tqdm(DataLoader(full_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                          pin_memory=True), desc='calculating class weight'):
            label_list.extend(label.numpy())

        return compute_class_weight(class_weight='balanced', classes=np.unique(label_list), y=label_list)


def create_file_path_list(root_path: str) -> List[str]:
    file_path_list = []
    for (root, dirs, files) in os.walk(root_path):
        if len(files) > 0:
            for file_name in files:
                file_path_list.append(root + os.sep + file_name)
    file_path_list.sort()

    return file_path_list
