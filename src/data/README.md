# Data Module

This folder contains utilities and classes for loading, preprocessing, and structuring spatiotemporal datasets for machine learning workflows, especially for transformer-based models.

## Contents

### 1. `spatiotemporal_transformer_grab_sensor_data.py`
- **Purpose:**
  - Provides functions for loading raw sensor data, scaling it, and extracting datapoints from 2D sensor arrays.
- **Key Functions:**
  - `load_data(path)`: Loads `.npy` files from a directory into a single array.
  - `scaling_data(data)`: Normalizes each channel of the data to [0, 1].
  - `extract_datapoints(dataset, x_indices, y_indices, separate_channels=True)`: Extracts specific points from the dataset based on x/y indices, optionally separating channels.

### 2. `spatiotemporal_transformer_dataset_generation.py`
- **Purpose:**
  - Contains classes and utilities for converting raw or preprocessed data into PyTorch datasets and dataloaders, with support for spectral features and time series splits.
- **Key Classes & Functions:**
  - `SimulationTimeSeries`: Handles time series construction, feature engineering, and train/val/test splitting.
  - `SimulationTorchDataset`: PyTorch `Dataset` for windowed time series data, ready for model input.
  - `DataModule`: PyTorch Lightning `LightningDataModule` for easy dataloader management.
  - `generate_spectral_welch(data, sf, nperseg, num_dominant_frequencies)`: Computes dominant frequencies using Welch’s method.

### 3. `__init__.py`
- **Purpose:**
  - Exposes a clean API for the data module. You can import all main utilities directly from `data`:
    ```python
    from data import (
        load_data, scaling_data, extract_datapoints,
        generate_spectral_welch, SimulationTimeSeries,
        SimulationTorchDataset, DataModule
    )
    ```

## Typical Usage

1. **Load and preprocess raw data:**
   ```python
   from data import extract_datapoints
   data_one, data_two = extract_datapoints(dataset, x_indices, y_indices, separate_channels=True)
   ```

2. **Prepare time series and dataloaders:**
   ```python
   from data import SimulationTimeSeries, SimulationTorchDataset, DataModule
   ts = SimulationTimeSeries(data=data_one, ...)
   dm = DataModule(dataset=SimulationTorchDataset, dataset_kwargs={'time_series': ts, ...}, ...)
   train_loader = dm.train_dataloader()
   ```

## Notes
- The code is designed for flexibility and research workflows, supporting both raw `.npy` sensor data and advanced time series feature engineering.
- All main classes and functions are exposed via the `data` package for easy import.