import argparse

import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
from S5.dataloaders.synthetics import ICLDataModule
import gpustat
from lightning.pytorch.loggers import WandbLogger

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
    torch.cuda.memory._record_memory_history(
        # keep a maximum 100,000 alloc/free events from before the snapshot
        max_entries=100000)

    args = get_args()
    
    global blueprint

    OmegaConf.register_new_resolver("eval", eval)
    blueprint = OmegaConf.load(args.blueprint_filepath)

    hyena = Hyena(lr=blueprint.learning_rate.base, **blueprint.model.config)
    dataset_obj = get_icl_synthetics_dataset(blueprint.dataset)

    wandb_logger = WandbLogger(project="hyena", log_model="all")
    trainer_args = dict(
        accumulate_grad_batches=blueprint.accumulate_grad_batches,
        log_every_n_steps=1,
        precision="bf16",
        logger=wandb_logger,
    )

    trainer = pl.Trainer(max_epochs=1, devices=1, **trainer_args)
    wandb_logger.watch(hyena)
    try:
        trainer.fit(model=hyena, train_dataloaders=dataset_obj.train_dataloader())
    except torch.cuda.OutOfMemoryError:
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump
        with open('snapshot.pickle', 'wb') as f:
            dump(snapshot, f)
        print(torch.cuda.memory_summary())
        print(gpustat.new_query())
        raise
    print(torch.cuda.memory_summary())
    metrics = pl.Trainer(
        devices=1, num_nodes=1, **trainer_args
    ).test(model=hyena, dataloaders=dataset_obj.test_dataloader())
    print(metrics)


if __name__ == "__main__":
    main()
