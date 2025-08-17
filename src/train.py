import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
import yaml
import os

from data.spatiotemporal_transformer_grab_sensor_data import extract_datapoints
from data.spatiotemporal_transformer_dataset_generation import SimulationTimeSeries, SimulationTorchDataset, DataModule
from additional_data_preprocessing.data_preprocessing import frequency_domain_augmentation, add_gaussian_noise
from trainer.spatiotemporal_performer_forecaster import SpatioTemporalPerformer_Forecaster

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    # Dataset loading
    dataset_path = config["dataset_path"]
    dataset = np.load(dataset_path)

    # Extract datapoints
    x_indices = config["x_indices"]
    y_indices = config["y_indices"]
    separate_channels = config["separate_channels"]
    data_one, data_two = extract_datapoints(dataset=dataset, x_indices=x_indices, y_indices=y_indices, separate_channels=separate_channels)

    which_dataset = config["which_dataset"]
    if which_dataset == "data_one":
        data = data_one
    elif which_dataset == "data_two":
        data = data_two
    else:
        raise ValueError("which_dataset must be 'data_one' or 'data_two'")

    # Train/val split
    split_idx = config["train_val_split_idx"]
    data_train = data[:split_idx, :]
    data_val = data[split_idx:, :]

    # Augmentation
    freq_noise_level = config["augmentation"]["freq_noise_level"]
    gauss_noise_level = config["augmentation"]["gauss_noise_level"]
    gauss_mean = config["augmentation"]["gauss_mean"]

    data_train_freq_augs = np.zeros_like(data_train)
    for i in range(data_train.shape[1]):
        data_train_freq_augs[:, i] = frequency_domain_augmentation(data_train[:, i], noise_level=freq_noise_level)
    data_train_augmented = add_gaussian_noise(data_train_freq_augs, mean=gauss_mean, noise_level=gauss_noise_level)

    # Concatenate train and val
    data_all = np.concatenate((data_train_augmented, data_val), axis=0)

    # Data module setup
    time_series = SimulationTimeSeries(
        data=data_all,
        start_time=82490,
        time_step_df=10,
        time_step_feature=1e-3,
        idx_start_train=0,
        idx_end_train=split_idx-1,
        idx_start_val=split_idx,
        idx_end_val=data.shape[0],
        have_test=False,
        num_dominant_frequencies=0,
        use_cycles=False,
    )

    data_module = DataModule(
        dataset=SimulationTorchDataset,
        dataset_kwargs={
            'time_series': time_series,
            'context_len': 80,
            'target_len': 4,
            'time_resolution': 1,
        },
        batch_size=4,
        num_workers=12,
        prediction=True,
        prediction_dataset="val",
        overfit=False
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    pred_loader = data_module.predict_dataloader()

    num_training_steps = config["num_training_steps"]
    model_args = config["model"].copy()
    model_args["num_training_steps"] = num_training_steps
    model = SpatioTemporalPerformer_Forecaster(**model_args)

    logger = pl.loggers.TensorBoardLogger('tb_log', name='test')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=20,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()