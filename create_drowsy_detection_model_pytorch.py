import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping
from multiprocessing import freeze_support
from utils.eye_dataset_utils import EyeDataModule, EyeDetModel, LitProgressBar


if __name__ == '__main__':
    freeze_support()

    wandb.login()
    wandb_logger = WandbLogger(project='drowsy_detection')

    bar = LitProgressBar()

    drowsy_det_model = EyeDetModel()
    data = EyeDataModule(train_dataset_desc=pd.read_csv('train_eye_shape_dataset.csv'),
                         val_dataset_desc=pd.read_csv('test_eye_shape_dataset.csv'),
                         batch_size=128, n_workers=2)

    early_stop_callback = EarlyStopping(monitor='train_loss', mode='min', verbose=True, min_delta=0.001, patience=200)
    trainer = pl.Trainer(accelerator='cpu', devices='auto', default_root_dir='weights/', logger=wandb_logger, benchmark=False, max_epochs=5,
                         log_every_n_steps=None, check_val_every_n_epoch=1, enable_progress_bar=True,
                         enable_model_summary=True, callbacks=[bar])

    trainer.fit(model=drowsy_det_model, datamodule=data)
