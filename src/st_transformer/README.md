# Spatio-Temporal Transformer (`st_transformer`)

This module implements a flexible, research-oriented spatio-temporal transformer for time-series forecasting, inspired by the SpaceTimeFormer model ([Grigsby et al., 2022](https://arxiv.org/abs/2201.00051)). The code is designed for simulation data (e.g., CFD), with special attention to embedding, tokenization, and efficient attention.

---

## Key Features

- **Spacetimeformer-style Embedding:**  
  - **Tokenization:** Flattens a multivariate window of shape (L, N) into L×N tokens, where each token represents a value y_{t,i} for variable i at timestep t.
  - **Time Features:** Calendar/time features are encoded using a Time2Vec layer, producing a mix of linear and sinusoidal components. These are concatenated with the scalar y_{t,i}.
  - **Projection & Sums:** The token is projected to `d_model` and summed with three learnable embeddings:
    1. **Position Embedding:** (learned or periodic Time2Vec)
    2. **Variable ID Embedding**
    3. **Given/Missing Flag Embedding:** Marks observed vs. masked values.

- **Flexible Time Handling:**  
  - For simulation data without timestamps (e.g., CFD), you can choose:
    - **No time embedding** (purely spatial)
    - **Spectral time embedding:** Use FFT/Welch to extract dominant modes and create sinusoidal time vectors.

- **Efficient Attention (Performer):**  
  - Implements Performer (FAVOR+) attention, replacing softmax with a kernel feature map Φ:
    - `softmax(QKᵀ)V ≈ Φ(Q) · (Φ(K)ᵀ V)`
    - Reduces attention complexity from O(L²) to O(L) for sequence length L.
    - Drop-in for multi-head attention, with unbiased/low-variance guarantees.

- **Highly Modular:**  
  - Encoder, decoder, embedding, and attention are all modular and extensible.
  - Includes normalization (GlobalNorm, RevIN), moving average/series decomposition, and a linear model baseline.

---

## File Overview

- **`spatiotemporalperformer.py`**  
  Main model class. Assembles embedding, encoder, decoder, and output head. Handles both teacher-forcing and autoregressive forecasting.

- **`spatiotemporal_transformer_embedding.py`**  
  Implements the Spacetimeformer-style embedding/tokenization logic, including Time2Vec, variable, and indicator embeddings.

- **`spatiotemporal_transformer_time2vec.py`**  
  Time2Vec embedding layer for flexible time/cycle encoding.

- **`spatiotemporal_transformer_attention.py`**  
  Performer (FAVOR+) attention and multi-head attention classes.

- **`spatiotemporal_transformer_encoder_part.py`**  
  Encoder block: stacks local/global attention, normalization, and feedforward layers.

- **`spatiotemporal_transformer_decoder_part.py`**  
  Decoder block: supports local/global self-attention and cross-attention, with normalization and feedforward layers.

- **`spatiotemporal_transformer_extralayers.py`**  
  Extra layers/utilities: normalization, convolutional blocks, masking, and tensor rearrangement helpers.

- **`global_norm.py`**  
  Global normalization layer for time series.

- **`revin.py`**  
  RevIN (Reversible Instance Normalization) for time series.

- **`moving_avg_series_decomp.py`**  
  Moving average and series decomposition modules for trend/seasonality separation.

- **`linear_model.py`**  
  Simple linear model baseline for forecasting.

- **`__init__.py`**  
  Exposes all main classes and utilities for easy import:
  ```python
  from st_transformer import (
      SpatioTemporalPerformer, Embedding, Encoder, Decoder,
      PerformerAttention, Time2Vec, GlobalNorm, RevIN, MovingAvg, SeriesDecomposition, LinearModel, ...
  )
  ```

---

## Example Usage

```python
from st_transformer import SpatioTemporalPerformer

model = SpatioTemporalPerformer(
    d_y_context=49, d_y_target=49, d_x=1, d_model=128, ...,
    time_embedding=True,  # or False for no time, or use spectral time
)
output = model(enc_x, enc_y, dec_x, dec_y)
```

---

## Notes

- The code is research-focused, modular, and designed for experimentation with spatio-temporal transformer architectures.
- The embedding and attention mechanisms are tailored for simulation data, but can be adapted for other spatio-temporal forecasting tasks.
- Inspired by SpaceTimeFormer, but adapted for simulation/CFD data and efficient Performer attention.