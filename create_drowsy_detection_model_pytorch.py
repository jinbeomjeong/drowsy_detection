import numpy as np
import pytorch_lightning as pl
from multiprocessing import freeze_support
from utils.ai_hub_eye_data_module import AiHubEyeDataModule, calculate_class_weight
from utils.eye_det_model import EyeDetModel
from pytorch_lightning.loggers import CSVLogger


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger(save_dir="logs", name="train_ai_hub", version=1)
    eye_img_root_path = 'd:\\eye_img_dataset'

    drowsy_det_model = EyeDetModel(learning_rate=0.005, class_weight=calculate_class_weight(eye_img_root_path=eye_img_root_path))

    data = AiHubEyeDataModule(bus_label_desc_path='label_data\\bus_result_20240602-235437.csv',
                              passenger_label_desc_path='label_data\\passenger_result_20240602-235514.csv',
                              taxi_label_desc_path='label_data\\taxi_result_20240602-235658.csv',
                              truck_label_desc_path='label_data\\truck_result_20240602-235807.csv',
                              target_driver_name='',
                              img_root_path='d:\\drowsy_dataset', eye_img_root_path=eye_img_root_path,
                              execute_prepare_data=False, batch_size=1500, n_workers=8)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', benchmark=False, max_epochs=100, logger=csv_logger, log_every_n_steps=1)
    trainer.fit(model=drowsy_det_model, datamodule=data)
