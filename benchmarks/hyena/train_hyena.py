import argparse

import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
from S5.dataloaders.synthetics import ICLDataModule

from hyena import Hyena


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint_filepath", required=True)
    args = parser.parse_args()
    
    return args


def get_icl_synthetics_dataset(cfg):
    assert cfg.type == "icl_synthetics"
    dataset_cfg = OmegaConf.to_container(cfg, resolve=True)
    del dataset_cfg["type"]
    dataset_obj = ICLDataModule(**dataset_cfg)
    dataset_obj.setup()

    return dataset_obj


def main():
    args = get_args()
    
    global blueprint

    OmegaConf.register_new_resolver("eval", eval)
    blueprint = OmegaConf.load(args.blueprint_filepath)

    hyena = Hyena(lr=blueprint.learning_rate.base, **blueprint.model.config)
    dataset_obj = get_icl_synthetics_dataset(blueprint.dataset)

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(model=hyena, train_dataloaders=dataset_obj.train_dataloader())
    metrics = pl.Trainer(
        devices=1, num_nodes=1, log_every_n_steps=1,
    ).test(model=hyena, dataloaders=dataset_obj.test_dataloader())
    print(metrics)


if __name__ == "__main__":
    main()
