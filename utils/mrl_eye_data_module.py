import os, cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms


class MrleyesDataset(Dataset):
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


class MrlEyeDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 16, n_workers: int = 8):
        super().__init__()
        self.__h_offset_scale: float = 0.08
        self.__v_offset_scale: float = 0.08
        self.__eye_img_files = list('')
        self.__eye_label_list = list('')

        self.__dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = n_workers
        self.__transform = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def prepare_data(self):
        for (path, dirs, files) in os.walk(self.__dataset_path):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    self.__eye_img_files.append(path + os.path.sep + file)

                    for i in range(4):
                        sep_idx = file.find('_')
                        file = file[sep_idx + 1:]

                    self.__eye_label_list.append(1-int(file[0]))

    def setup(self, stage=None):
        self.__train_dataset = MrleyesDataset(img_path_list=self.__eye_img_files, label_list=self.__eye_label_list,
                                              transform=self.__transform)
        self.__train_dataset, self.__val_dataset = random_split(self.__train_dataset, lengths=[0.6, 0.4])
        self.__val_dataset, self.__test_dataset = random_split(self.__train_dataset, lengths=[0.5, 0.5])

    def train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.__val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.__test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)
