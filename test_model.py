import pytorch_lightning as pl
from multiprocessing import freeze_support
from utils.ai_hub_eye_data_module import AiHubEyeDataModule
from utils.eye_det_model import EyeDetModel
from pytorch_lightning.loggers import CSVLogger


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger(save_dir="logs", name="test_ai_hub", version=1)

    model = EyeDetModel.load_from_checkpoint(checkpoint_path="logs\\train_ai_hub\\version_1\\checkpoints\\epoch=49-step=700.ckpt")
    data = AiHubEyeDataModule(bus_label_desc_path='label_data\\bus_result_20240602-235437.csv',
                              passenger_label_desc_path='label_data\\passenger_result_20240602-235514.csv',
                              taxi_label_desc_path='label_data\\taxi_result_20240602-235658.csv',
                              truck_label_desc_path='label_data\\truck_result_20240602-235807.csv',
                              img_root_path='d:\\drowsy_dataset', eye_img_root_path='d:\\eye_img_dataset',
                              target_driver_name='', execute_prepare_data=False, batch_size=1, n_workers=4)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', logger=csv_logger, benchmark=False, log_every_n_steps=1)
    trainer.test(model=model, datamodule=data)
