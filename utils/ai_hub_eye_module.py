import os, cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class AiHubEyesDataset(Dataset):
    def __init__(self, img_path_list: list, label_list: list, transform: transforms):
        self.__img_path_list = img_path_list
        self.__label_list = label_list
        self.__transform = transform

    def __len__(self):
        return len(self.__img_path_list)

    def __getitem__(self, idx):
        eye_img = cv2.imread(self.__img_path_list[idx])
        eye_img = self.__transform(eye_img)

        return eye_img, self.__label_list[idx]


class AiHubEyeDataModule(pl.LightningDataModule):
    def __init__(self, bus_label_desc_path: str, passenger_label_desc_path: str, taxi_label_desc_path: str, truck_label_desc_path: str,
                 img_root_path: str, output_eye_img_root_path: str,  execute_prepare_data=False, batch_size: int = 16, n_workers: int = 8):
        super().__init__()
        self.__h_offset_scale: float = 0.08
        self.__v_offset_scale: float = 0.08
        self.__eye_img_files = list()
        self.__left_eye_pos = list()
        self.__right_eye_pos = list()
        self.__left_eye_label_list = list()
        self.__right_eye_label_list = list()

        self.__bus_label_desc_path = bus_label_desc_path
        self.__passenger_label_desc_path = passenger_label_desc_path
        self.__taxi_label_desc_path = taxi_label_desc_path
        self.__truck_label_desc_path = truck_label_desc_path
        self.__img_root_path = img_root_path
        self.__output_eye_img_root_path = output_eye_img_root_path
        self.__execute_prepare_data = execute_prepare_data

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
            bus_label_data = pd.read_csv(self.__bus_label_desc_path)
            bus_driver = pd.DataFrame(['bus']*bus_label_data.shape[0], columns=['driver_type'])
            bus_label_data = pd.concat([bus_label_data, bus_driver], axis=1)

            passenger_label_data = pd.read_csv(self.__passenger_label_desc_path)
            passenger_driver = pd.DataFrame(['passenger']*passenger_label_data.shape[0], columns=['driver_type'])
            passenger_label_data = pd.concat([passenger_label_data, passenger_driver], axis=1)

            taxi_label_data = pd.read_csv(self.__taxi_label_desc_path)
            taxi_driver = pd.DataFrame(['taxi']*taxi_label_data.shape[0], columns=['driver_type'])
            taxi_label_data = pd.concat([taxi_label_data, taxi_driver], axis=1)

            truck_label_data = pd.read_csv(self.__truck_label_desc_path)
            truck_driver = pd.DataFrame(['truck']*truck_label_data.shape[0], columns=['driver_type'])
            truck_label_data = pd.concat([truck_label_data, truck_driver], axis=1)

            bus_train_set, bus_val_set = train_test_split(bus_label_data, test_size=0.4, random_state=0)
            bus_val_set, bus_test_set = train_test_split(bus_val_set, test_size=0.5, random_state=0)

            passenger_train_set, passenger_val_set = train_test_split(passenger_label_data, test_size=0.4, random_state=0)
            passenger_val_set, passenger_test_set = train_test_split(passenger_val_set, test_size=0.5, random_state=0)

            taxi_train_set, taxi_val_set = train_test_split(taxi_label_data, test_size=0.4, random_state=0)
            taxi_val_set, taxi_test_set = train_test_split(taxi_val_set, test_size=0.5, random_state=0)

            truck_train_set, truck_val_set = train_test_split(truck_label_data, test_size=0.4, random_state=0)
            truck_val_set, truck_test_set = train_test_split(truck_val_set, test_size=0.5, random_state=0)

            train_set = pd.concat([bus_train_set, passenger_train_set, taxi_train_set, truck_train_set])
            train_set.reset_index(drop=True, inplace=True)

            val_set = pd.concat([bus_val_set, passenger_val_set, taxi_val_set, truck_val_set])
            val_set.reset_index(drop=True, inplace=True)

            for i in tqdm(range(train_set.shape[0]), desc='preparing train dataset'):
                find_img = False

                target_file_name = train_set.loc[i, 'file_name']
                eye_pos = train_set.loc[i, ['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
                eye_label = train_set.loc[i, 'opened']
                eye_label_name = 'open' if eye_label == 1 else 'close'
                driver_type = train_set.loc[i, 'driver_type']

                for (path, dirs, files) in os.walk(self.__img_root_path):
                    for file_name in files:
                        if file_name == target_file_name:
                            img = Image.open(path + os.path.sep + file_name)
                            eye_img = img.crop(tuple(eye_pos))
                            eye_img.save(self.__output_eye_img_root_path + os.path.sep + 'train' + os.path.sep +
                                         eye_label_name + os.path.sep + os.path.splitext(file_name)[0] + '_' + str(i) + '_' + driver_type + '.jpg', format='JPEG')
                            find_img = True
                            break

                    if find_img:
                        break

            for i in tqdm(range(val_set.shape[0]), desc='preparing val dataset'):
                find_img = False
                target_file_name = val_set.loc[i, 'file_name']
                eye_pos = val_set.loc[i, ['eye_pt1_x_pos', 'eye_pt1_y_pos', 'eye_pt2_x_pos', 'eye_pt2_y_pos']].to_numpy(dtype=np.uint16)
                eye_label = val_set.loc[i, 'opened']
                eye_label_name = 'open' if eye_label == 1 else 'close'
                driver_type = val_set.loc[i, 'driver_type']

                for (path, dirs, files) in os.walk(self.__img_root_path):
                    for file_name in files:
                        if file_name == target_file_name:
                            img = Image.open(path + os.path.sep + file_name)
                            eye_img = img.crop(tuple(eye_pos))
                            eye_img.save(self.__output_eye_img_root_path + os.path.sep + 'val' + os.path.sep +
                                         eye_label_name + os.path.sep + os.path.splitext(file_name)[0] + '_' + str(i) + '_' + driver_type + '.jpg', format='JPEG')
                            find_img = True
                            break

                    if find_img:
                        break

    def setup(self, stage=None):
        self.__train_dataset = ImageFolder(root=self.__output_eye_img_root_path + os.sep + 'train', transform=self.__transform)
        self.__val_dataset = ImageFolder(root=self.__output_eye_img_root_path + os.sep + 'val', transform=self.__transform)

    def train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.__val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.__test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)