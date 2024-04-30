import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping
from multiprocessing import freeze_support
from utils.eye_dataset_utils import EyeDataModule, EyeDetModel, LitProgressBar


if __name__ == '__main__':
    freeze_support()

    device_stats = DeviceStatsMonitor()
    bar = LitProgressBar()

    drowsy_det_model = EyeDetModel()
    data = EyeDataModule(train_dataset_desc=pd.read_csv('train_eye_shape_dataset.csv'),
                         val_dataset_desc=pd.read_csv('test_eye_shape_dataset.csv'),
                         batch_size=2048, n_workers=4)

    early_stop_callback = EarlyStopping(monitor='train_loss', mode='min', verbose=True, min_delta=0.001, patience=200)
    trainer = pl.Trainer(accelerator='gpu', devices='auto', benchmark=False, max_epochs=50, log_every_n_steps=None,
                         check_val_every_n_epoch=1, enable_progress_bar=True, enable_model_summary=True,
                         callbacks=[bar])

    trainer.fit(model=drowsy_det_model, datamodule=data)
