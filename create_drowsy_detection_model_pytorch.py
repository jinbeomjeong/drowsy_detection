import pytorch_lightning as pl
from multiprocessing import freeze_support
from utils.ai_hub_eye_data_module import AiHubEyeDataModule
from utils.eye_det_model import EyeDetModel
from pytorch_lightning.loggers import CSVLogger


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger(save_dir="logs", name="train_ai_hub", version=1)
    eye_img_root_path = 'd:\\eye_img_dataset'

    data = AiHubEyeDataModule(train_eye_label_desc_path="label_data\\train_eye_label.csv",
                              val_eye_label_desc_path="label_data\\val_eye_label.csv",
                              eye_img_root_path=eye_img_root_path,
                              target_driver_name='', execute_prepare_data=False, batch_size=2000, n_workers=8)

    drowsy_det_model = EyeDetModel(learning_rate=0.05, class_weight=data.calculate_class_weight())

    trainer = pl.Trainer(accelerator='gpu', devices='auto', benchmark=False, max_epochs=100, logger=csv_logger, log_every_n_steps=1)
    trainer.fit(model=drowsy_det_model, datamodule=data)
