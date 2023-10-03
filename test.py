import os
import time
import hydra
import torch
import scipy.io
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, loggers

from utils.utilities import load_dataset, divide_patches, get_performance
from dataloaders.dataset import MyDataSet
from models.BDQM import MyNet


class MyModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config

        self.net = MyNet(config)
        self.criterion = torch.nn.L1Loss()

        self.test_img_path = None
        self.test_data = None
        self.mos_list = []
        self.pred_list = []
        self.test_time = []

    # =========================================load dataset=========================================

    def prepare_data(self):
        self.test_img_path = self.cfg.test_dataset.image_path
        self.test_data = load_dataset(self.cfg.test_dataset.test_data_path)

    def test_dataloader(self):
        test_dataset = MyDataSet(
            img_path=self.test_img_path,
            split_data=self.test_data,
            use_augmentation=False,
            resize_size=self.cfg.test_dataset.resize_size,
            iscrop=self.cfg.test_dataset.iscrop,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.cfg.test_batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.cfg.num_workers)

    def forward(self, x):
        return self.net(x)

    # =========================================step operation=========================================

    def _step(self, batch):
        image, mos = batch
        image_patches = divide_patches(image, patch_size=self.cfg.patch_size, state='test')

        start_time = time.time()
        pred, _ = self.net(image_patches)
        end_time = time.time()
        self.test_time.append(end_time - start_time)

        loss = self.criterion(mos, pred)
        # self.log(f"test/loss", loss)

        return loss, mos, pred

    def test_step(self, batch, batch_idx):
        loss, mos, pred = self._step(batch)
        return {"loss": loss, "mos": mos, "pred": pred}

    # =========================================epoch operation=========================================

    def _epoch(self, outputs):
        self.mos_list.clear()
        self.pred_list.clear()
        for step_out in outputs:
            self.mos_list.append(step_out['mos'].item())
            self.pred_list.append(step_out['pred'].item())

    def test_epoch_end(self, test_step_outputs):
        self._epoch(test_step_outputs)
        srocc, krocc, plcc = get_performance(self.mos_list, self.pred_list)
        self.log(f"test/srocc", srocc, on_epoch=True, on_step=False)
        self.log(f"test/krocc", krocc, on_epoch=True, on_step=False)
        self.log(f"test/plcc", plcc, on_epoch=True, on_step=False)

        scipy.io.savemat(os.path.join(self.cfg.save_path, 'result.mat'), {'mos': self.mos_list, 'pred': self.pred_list})
        time_log = f'Total time: {np.sum(self.test_time)}\nTotal images: {len(self.test_time)}\nAverage time: {np.mean(self.test_time)}'
        print(f'\n{time_log}')


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.initialization_seed)

    logger = loggers.TensorBoardLogger("logs", name=cfg.task_name)
    cfg.save_path = logger.log_dir

    model = MyModule(cfg)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=logger
    )

    if cfg.test_only:
        trainer.test(model, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
