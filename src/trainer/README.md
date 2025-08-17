# SpatioTemporal Performer Forecaster (`spatiotemporal_performer_forecaster.py`)

This module provides the high-level training and forecasting interface for spatio-temporal transformer models, built on top of PyTorch Lightning. It is designed for research and experimentation with advanced time-series forecasting architectures, especially for simulation and CFD data.

---

## Key Features

- **Modular Forecaster Base:**
  - `Forecaster` is an abstract base class (inherits from `pl.LightningModule`) that defines the training, validation, and test steps, loss computation, and metric logging for time-series forecasting.
  - Supports multiple loss functions (MSE, MAE, Huber) and flexible masking for missing/null values.

- **Normalization & Decomposition:**
  - Supports global normalization (`GlobalNorm`), reversible instance normalization (`RevIN`), and no normalization.
  - Optional seasonal-trend decomposition using moving averages (`SeriesDecomposition`).

- **Linear Baseline:**
  - Optionally adds a linear model baseline to the forecast, for hybrid deep+linear forecasting.

- **Flexible Training Strategies:**
  - Handles both teacher-forcing and autoregressive (scheduled sampling) training.
  - Implements scheduled sampling with inverse sigmoid decay for teacher-forcing ratio.
  - Supports both one-shot and step-by-step autoregressive forecasting.

- **Optimizer & Scheduler:**
  - AdamW optimizer with both warmup and plateau learning rate schedulers.

- **Metrics:**
  - Logs MAE, MSE, and loss for both training and validation.

---

## Main Classes

- **`Forecaster`**
  - Abstract base class for time-series forecasting models.
  - Handles normalization, decomposition, loss computation, and metric logging.
  - Requires implementation of `forward_model_pass` (the actual model forward logic).

- **`SpatioTemporalPerformer_Forecaster`**
  - Concrete implementation that wraps a `SpatioTemporalPerformer` model (from `st_transformer`).
  - Handles all model hyperparameters, scheduled sampling, and orchestrates the full training loop.
  - Supports both teacher-forcing and autoregressive forecasting.

---

## Example Usage

```python
from trainer.spatiotemporal_performer_forecaster import SpatioTemporalPerformer_Forecaster

model = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=1,
    d_model=128,
    # ... other hyperparameters ...
    use_seasonal_decomp=True,
    use_revin=True,
    use_global_norm=False,
    num_training_steps=30000,
    autoregressive_training=False,
)
```

---

## Notes
- This module is designed for research and experimentation, not production.
- It is highly modular: you can swap in different normalization, decomposition, or model architectures.
- The code is compatible with PyTorch Lightning's Trainer, callbacks, and logging ecosystem.
- Inspired by recent advances in spatio-temporal transformers and efficient attention for long-range forecasting.