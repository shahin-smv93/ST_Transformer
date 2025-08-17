# Spatio-Temporal Transformer for Simulation Data Forecasting

> **Note:** The code is fully functional and ready to use. Results based on our CFD data (along with the data) will be added after the related paper is published.

This project implements a modular, research-focused spatio-temporal transformer for time-series forecasting, inspired by [SpaceTimeFormer (Grigsby et al., 2022)](https://arxiv.org/abs/2201.00051) and enhanced with Performer (FAVOR+) efficient attention. The code is designed for simulation data (e.g., CFD), but is flexible for other spatio-temporal forecasting tasks.

---

## Project Structure

```
project-root/
│
├── dataset/                        # Place your Dataset.npy here
│   └── Dataset.npy
│
├── src/
│   ├── data/                       # Data loading, extraction, and dataset utilities
│   ├── st_transformer/             # Spatio-temporal transformer model and components
│   ├── trainer/                    # High-level training and forecasting interface
│   └── train.py                    # Main training script
│
├── config.yaml                     # All experiment and model configuration
├── requirements.txt
└── README.md                       # (This file)
```

---

## Key Components

### Data Module (`src/data/`)
- **Purpose:** Load, preprocess, and structure spatiotemporal datasets.
- **Highlights:**
  - Extracts datapoints from 2D sensor arrays.
  - Supports normalization and flexible dataset construction.
  - Provides PyTorch and PyTorch Lightning dataset/dataloader classes.

### Spatio-Temporal Transformer (`src/st_transformer/`)
- **Purpose:** Implements a modular transformer for spatio-temporal forecasting.
- **Highlights:**
  - **Spacetimeformer-style Embedding:** Flattens (L, N) windows, encodes time with Time2Vec, and adds variable/position/indicator embeddings.
  - **Flexible Time Handling:** Supports no time, or spectral time embedding (FFT/Welch).
  - **Performer Attention:** Efficient FAVOR+ attention for long sequences.
  - **Modular:** Encoder, decoder, normalization, and linear/trend baselines.

### Trainer (`src/trainer/`)
- **Purpose:** High-level training and forecasting interface using PyTorch Lightning.
- **Highlights:**
  - Modular `Forecaster` base class.
  - Handles normalization, decomposition, and hybrid deep+linear forecasting.
  - Supports teacher-forcing, scheduled sampling, and autoregressive forecasting.
  - Integrated optimizer, scheduler, and metric logging.

---

## Configuration

All experiment and model parameters are set in `config.yaml`:

```yaml
dataset_path: "dataset/Dataset.npy"
x_indices: [4, 8, 15, 20, 27, 38, 51]
y_indices: [3, 7, 14, 24, 50, 85, 140]
separate_channels: true
which_dataset: "data_one"  # or "data_two"
train_val_split_idx: 6084

augmentation:
  freq_noise_level: 0.05
  gauss_noise_level: 10
  gauss_mean: 0

num_training_steps: 30000

model:
  d_y_context: 49
  d_y_target: 49
  d_x: 1
  d_model: 128
  # ... (see config.yaml for all hyperparameters)
```

---

## Training Workflow

1. **Edit `config.yaml`** to set your dataset path, indices, augmentation, and model hyperparameters.
2. **Run the training script:**
   ```bash
   cd src
   python train.py
   ```
   - This will:
     - Load and preprocess your data.
     - Set up the model and dataloaders.
     - Train the model using PyTorch Lightning, with TensorBoard logging and checkpointing.

---

## Example: Using the Model in Code

```python
from trainer.spatiotemporal_performer_forecaster import SpatioTemporalPerformer_Forecaster

model = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=1,
    d_model=128,
    # ... other hyperparameters from config.yaml ...
)
```

---

## Notes

- The codebase is modular and research-focused, designed for easy experimentation and extension.
- The transformer architecture and embedding/tokenization are inspired by SpaceTimeFormer, but adapted for simulation/CFD data and efficient Performer attention.
- All main classes and functions are exposed via their respective packages for easy import.
- For more details, see the README files in each submodule (`src/data/`, `src/st_transformer/`, `src/trainer/`).

---

## References

- **SpaceTimeFormer:**
  - Grigsby, A., Xu, J., Wu, Z., & Lipton, Z. C. (2022). [Long-Range Transformers for Dynamic Spatiotemporal Forecasting](https://arxiv.org/abs/2201.00051). arXiv:2201.00051.

- **Performer Attention:**
  - Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., & Weller, A. (2021). [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794). arXiv:2009.14794.

---

**For questions or contributions, please open an issue or pull request!**