import sys, torch, torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import mobilenet_v2
from pytorch_lightning.callbacks import TQDMProgressBar


class EyeDetModel(pl.LightningModule):
    def __init__(self, class_weight, learning_rate: float = 0.001):
        super().__init__()
        self.__learning_rate = learning_rate
        self.__model = mobilenet_v2(weights=None)
        self.__model.classifier[1] = nn.Sequential(nn.Dropout(0.5),
                                                   nn.Linear(self.__model.classifier[1].in_features, out_features=2))

        self.__criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight), reduction='mean')
        self.__accuracy_score = torchmetrics.Accuracy(task='binary')
        self.__precision_score = torchmetrics.Precision(task='binary')
        self.__recall_score = torchmetrics.Recall(task='binary')
        self.__f1_score = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        return self.__model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.__criterion(pred, label)

        pred_result = torch.argmax(pred, dim=1)

        self.log('train_loss', loss, sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name='train_accuracy', value=self.__accuracy_score(pred_result, label), sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name='train_precision', value=self.__precision_score(pred_result, label), sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name='train_recall', value=self.__recall_score(pred_result, label), sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name='train_f1', value=self.__f1_score(pred_result, label), sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.__criterion(pred, label)

        pred_result = torch.argmax(pred, dim=1)

        self.log(name='val_loss', value=loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_accuracy', value=self.__accuracy_score(pred_result, label), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_precision', value=self.__precision_score(pred_result, label), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_recall', value=self.__recall_score(pred_result, label), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log(name='val_f1', value=self.__f1_score(pred_result, label), sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        pred_result = torch.argmax(pred, dim=1)

        self.log(name='label', value=label, sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name='pred', value=pred_result, sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__learning_rate)

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
