import cv2, torch, sys
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import torchmetrics


class EyeDataset(Dataset):
    def __init__(self, dataset_desc: pd.DataFrame, transform: transforms):
        self.h_offset_scale = 0.05
        self.v_offset_scale = 0.05

        self.__dataset_desc = dataset_desc
        self.__transform = transform
        self.__img_file_name = ''
        self.__eye_position = list()
        self.__img = np.array([])
        self.__eye_img = np.array([])

    def __len__(self):
        return self.__dataset_desc.shape[0]

    def __getitem__(self, idx):
        self.__img_file_name = self.__dataset_desc.loc[idx, 'file_name']
        self.__img = cv2.imread(self.__img_file_name)

        self.__eye_position = [self.__dataset_desc.loc[idx, 'target_left_eye_pt1_x_pos'],
                               self.__dataset_desc.loc[idx, 'target_left_eye_pt1_y_pos'],
                               self.__dataset_desc.loc[idx, 'target_left_eye_pt2_x_pos'],
                               self.__dataset_desc.loc[idx, 'target_left_eye_pt2_y_pos']]

        self.__eye_position[0] = int(self.__eye_position[0] - (self.__eye_position[0] * self.h_offset_scale))
        self.__eye_position[1] = int(self.__eye_position[1] - (self.__eye_position[1] * self.v_offset_scale))
        self.__eye_position[2] = int(self.__eye_position[2] + (self.__eye_position[2] * self.h_offset_scale))
        self.__eye_position[3] = int(self.__eye_position[3] + (self.__eye_position[3] * self.v_offset_scale))

        self.__eye_img = self.__img[self.__eye_position[1]:self.__eye_position[3],
                         self.__eye_position[0]:self.__eye_position[2]]

        if self.__transform:
            self.__eye_img = Image.fromarray(self.__eye_img)
            self.__eye_img = self.__transform(self.__eye_img)

        return self.__eye_img, self.__dataset_desc.loc[idx, 'left_eye_opened'].item()


class EyeDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset_desc: pd.DataFrame, val_dataset_desc: pd.DataFrame, batch_size: int = 16,
                 n_workers: int = 8):
        super().__init__()
        self.__train_dataset_desc = train_dataset_desc
        self.__val_dataset_desc = val_dataset_desc
        self.batch_size = batch_size
        self.num_workers = n_workers
        self.__transform = transforms.Compose([transforms.Resize((64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    def setup(self, stage=None):
        self.__train_dataset = EyeDataset(self.__train_dataset_desc, transform=self.__transform)
        self.__val_dataset = EyeDataset(self.__val_dataset_desc, transform=self.__transform)

    def train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.__val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)


class EyeDetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.__model = models.resnet18(weights=None)
        self.__model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.__model.fc = torch.nn.Linear(in_features=self.__model.fc.in_features, out_features=2)
        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__accuracy_score = torchmetrics.Accuracy(task='binary')
        self.__precision_score = torchmetrics.Precision(task='binary')
        self.__recall_score = torchmetrics.Recall(task='binary')
        self.__f1_score = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        return self.__model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label
        pred = self.forward(img)
        loss = self.__criterion(pred, label)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label
        pred = self.forward(img)
        loss = self.__criterion(pred, label)

        pred_result = torch.argmax(pred, dim=1)

        self.log(name='val_loss', value=loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_accuracy', value=self.__accuracy_score(pred_result, label), on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_precision', value=self.__precision_score(pred_result, label), on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_recall', value=self.__recall_score(pred_result, label), on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_f1', value=self.__f1_score(pred_result, label), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.001)

        return optimizer


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()

        if not sys.stdout.isatty():
            bar.disable = True

        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()

        if not sys.stdout.isatty():
            bar.disable = True

        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()

        if not sys.stdout.isatty():
            bar.disable = True
        return bar
