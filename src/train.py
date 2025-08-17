

### data loading and preprocessing ###

from spatiotemporal_transformer_grab_sensor_data import *

dataset_path = '/content/drive/MyDrive/Spatiotemporal_Transformer/Dataset.npy'
dataset = np.load(dataset_path)

def frequency_domain_augmentation(signal, noise_level=0.05):
    fft_vals = np.fft.fft(signal)
    amplitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    # perturb amplitude and phase with random noise
    amplitude_perturbed = amplitude * (1 + noise_level * np.random.randn(*amplitude.shape))
    phase_perturbed = phase + noise_level * np.random.randn(*phase.shape)

    fft_aug = amplitude_perturbed * np.exp(1j * phase_perturbed)
    signal_aug = np.fft.ifft(fft_aug).real
    return signal_aug

def add_gaussian_noise(data, mean=0, noise_level=50):
    std_dev = np.std(data, axis=0)
    noise = np.random.normal(mean, scale=(noise_level / 100) * std_dev, size=data.shape)
    noisy_data = data + noise

    return noisy_data

def compute_lag1_autocorrelation(data):
    autocorrs = []
    T, num_features = data.shape
    for i in range(num_features):
        sensor_data = data[:, i]
        corr = np.corrcoef(sensor_data[:-1], sensor_data[1:])[0, 1]
        autocorrs.append(corr)
    return autocorrs

def compute_global_mean_std(data):
    global_mean = np.mean(data, axis=0)
    global_std = np.std(data, axis=0)
    return global_mean, global_std

data_one, data_two = extract_datapoints(dataset=dataset,
                                        x_indices=[4, 8, 15, 20, 27, 38, 51],
                                        y_indices=[3, 7, 14, 24, 50, 85, 140],
                                        separate_channels=True)

uprime = data_one - np.mean(data_one, axis=0)
vprime = data_two - np.mean(data_two, axis=0)

data_one, data_two = extract_datapoints(dataset=dataset,
                                        x_indices=[4, 8, 15],
                                        y_indices=[3, 7, 14],
                                        separate_channels=True)

data_train_one = data_two[:6084, :]
data_val_one = data_two[6084:, :]

import matplotlib.pyplot as plt
data = data_train_one
window = 50

for i in range(data.shape[1]):
    feature = data[:, i]
    rolling_mean = np.convolve(feature, np.ones(window)/window, mode='valid')
    rolling_std = np.array([feature[max(0, t-window):t].std() for t in range(1, len(feature)+1)])

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(rolling_mean)
    plt.title(f'Rolling Mean for feature {i}')
    plt.subplot(1,2,2)
    plt.plot(rolling_std[window-1:])  # aligning with rolling mean
    plt.title(f'Rolling Std for feature {i}')
    plt.show()

from scipy.stats import ks_2samp

# Split data into two segments
split_index = int(0.5 * data.shape[0])
segment1 = data[:split_index, :]
segment2 = data[split_index:, :]

for i in range(data.shape[1]):
    stat, p_value = ks_2samp(segment1[:, i], segment2[:, i])
    print(f"Feature {i}: KS statistic = {stat:.4f}, p-value = {p_value:.4f}")

from scipy.stats import entropy, wasserstein_distance

for i in range(data.shape[1]):
    hist1, bin_edges = np.histogram(segment1[:, i], bins=50, density=True)
    hist2, _ = np.histogram(segment2[:, i], bins=bin_edges, density=True)
    # Add a small constant to avoid division by zero
    hist1 += 1e-8
    hist2 += 1e-8

    kl_div = entropy(hist1, hist2)
    wass_dist = wasserstein_distance(segment1[:, i], segment2[:, i])
    print(f"Feature {i}: KL divergence = {kl_div:.4f}, Wasserstein distance = {wass_dist:.4f}")



# 2 steps augmentation on training data
data_train_one_freq_augs = np.zeros_like(data_train_one)
for i in range(data_train_one.shape[1]):
    data_train_one_freq_augs[:, i] = frequency_domain_augmentation(data_train_one[:, i], noise_level=0.05)

data_one_train_augmented = add_gaussian_noise(data_train_one_freq_augs, mean=0, noise_level=10)

import matplotlib.pyplot as plt

x = np.arange(data_one_train_augmented.shape[0])
plt.figure(figsize=(12, 6))
plt.plot(x, data_one_train_augmented[:, 2], label='augmented', color='blue')
plt.plot(x, data_train_one[:, 2], label='original', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# diff
data_one_train_diff = np.diff(data_one_train_augmented, axis=0)
data_one_val_diff = np.diff(data_val_one, axis=0)
data_one_train_diff.shape, data_one_val_diff.shape

# scale data between 0 and 1
x = data_one_train_diff
data_one_train_diff_scaled = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

y = data_one_val_diff
data_one_val_diff_scaled = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

# scale between 0 and 1
x = data_one_train_augmented
data_one_train_scaled = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

y = data_val_one
data_one_val_scaled = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

np.min(data_one_train_scaled)

autocorr = compute_lag1_autocorrelation(data_one_val_diff_scaled)
print(autocorr)

data_one_train_diff

global_mean, global_std = compute_global_mean_std(data_train_one)

val_noisy = add_gaussian_noise(data_val_one, mean=0, noise_level=20)
#val_noisy[200:300] = 10 * np.random.rand(100, 49) * val_noisy[200:300]
data_one = np.concatenate((data_one_train_augmented, val_noisy), axis=0)

val_noisy = np.random.rand(*data_val_one.shape)
val_noisy[:300] = 3.5423 * val_noisy[:300]
val_noisy[300:800] = 2.6543 * val_noisy[300:800]
val_noisy[800:] = 4.87687 * val_noisy[800:]
data_one = np.concatenate((data_one_train_augmented, val_noisy), axis=0)
data_one.shape

x = np.arange(val_noisy.shape[0])
plt.figure(figsize=(12, 6))
plt.plot(x, data_val_one[:, -10], label='noisy', color='blue')
plt.plot(x, val_noisy[:, -10], label='original', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 2 steps augmentation on training data
data_val_one_freq_augs = np.zeros_like(data_val_one)
for i in range(data_train_one.shape[1]):
    data_val_one_freq_augs[:, i] = frequency_domain_augmentation(data_val_one[:, i], noise_level=0.05)

data_one_val_augmented = add_gaussian_noise(data_val_one_freq_augs, mean=0, noise_level=10)

data_one = np.concatenate((data_one_train_augmented, data_val_one), axis=0)

data_one.shape

### data module ###

from spatiotemporal_transformer_dataset_generation import *

time_series = SimulationTimeSeries(
    data=data_one,
    start_time=82490,
    time_step_df=10,
    time_step_feature=1e-3,
    idx_start_train=0,
    idx_end_train=6083,
    idx_start_val=6090,
    idx_end_val=7604,
    have_test=False,
    num_dominant_frequencies=0,
    use_cycles=False,
)

# DataModule for df_one
data_module_one = DataModule(
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

train_loader_one = data_module_one.train_dataloader()
val_loader_one = data_module_one.val_dataloader()
pred_loader_one = data_module_one.predict_dataloader()

### model training ###

batch = next(iter(val_loader_one))
context_x, context_y, target_x, target_y = batch

print(f"context_x shape: {context_x.shape}")
print(f"context_y shape: {context_y.shape}")

count = 0
for idx, batch in enumerate(train_loader_one):
    count = count + 1

count

num_trainin_steps = 1500 * 20

model = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=20,
    d_model=768,
    d_q_k=96,
    d_v=96,
    n_heads=6,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=1536,
    time_emb_dim=20,
    activation='gelu',
    initial_lr=1e-6,
    loss='mse',
    l2_regul_factor=1e-4,
    verbose=True,
    start_len=0,
    max_seq_len=22,
    normalization_type='batchnorm',
    data_dropout_embedding=0.0,
    dropout_ff=0.2,
    redraw_interval=500,
    use_seasonal_decomp=True,
    use_revin=True,
    use_global_norm=False,
    global_mean=global_mean,
    global_std=global_std,
    linear_window=10,
    linear_shared_weights=True,
    time_embedding=True,
    indicator_embedding=True,
    decay_factor=0.1,
    warmup_steps=200,
    num_training_steps=num_trainin_steps
)

model = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=20,
    d_model=128,
    d_q_k=32,
    d_v=32,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=256,
    time_emb_dim=10,
    activation='relu',
    initial_lr=5e-4,
    loss='huber',
    l2_regul_factor=1e-4,
    verbose=True,
    start_len=0,
    max_seq_len=22,
    normalization_type='batchnorm',
    data_dropout_embedding=0.2,
    dropout_ff=0.2,
    redraw_interval=100,
    use_seasonal_decomp=True,
    use_revin=False,
    use_global_norm=False,
    global_mean=None,
    global_std=None,
    linear_window=10,
    linear_shared_weights=True,
    time_embedding=True,
    indicator_embedding=True,
    decay_factor=0.1,
    warmup_steps=200,
    num_training_steps=num_trainin_steps,
    autoregressive_training=True
)

model = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=1,
    d_model=128,
    d_q_k=16,
    d_v=16,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=256,
    time_emb_dim=2,
    activation='relu',
    initial_lr=1e-4,
    loss='mse',
    l2_regul_factor=1e-4,
    verbose=True,
    start_len=0,
    max_seq_len=84,
    normalization_type='batchnorm',
    data_dropout_embedding=0.0,
    dropout_ff=0.1,
    redraw_interval=500,
    use_seasonal_decomp=True,
    use_revin=True,
    use_global_norm=False,
    global_mean=None,
    global_std=None,
    linear_window=20,
    linear_shared_weights=True,
    time_embedding=True,
    indicator_embedding=True,
    decay_factor=0.1,
    warmup_steps=4000,
    num_training_steps=num_trainin_steps,
    autoregressive_training=False,
    scheduled_sampling_k=5000.0
)

logger = pl.loggers.TensorBoardLogger('tb_log', name='test')

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val/loss',
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)

from pytorch_lightning.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    max_epochs=20,
    accelerator="auto",
    devices="auto",
    #accumulate_grad_batches=2,
    #gradient_clip_val=1,
    #detect_anomaly=True,
)

CUDA_LAUNCH_BLOCKING=1

!pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/

!pwd

!pip freeze > requirements.txt

#torch.autograd.set_detect_anomaly(True)
trainer.fit(model, datamodule=data_module_one)

context_x, _, target_x, _ = next(iter(val_loader_one))
print(context_x.shape)
print(target_x.shape)

g_mean = np.expand_dims(global_mean, axis=0)
g_std = np.expand_dims(global_std, axis=0)

# g_mean = np.expand_dims(g_mean, axis=0)
# g_std = np.expand_dims(g_std, axis=0)

g_mean.shape, g_std.shape

### Re-instantiate for inference

model_no_time = SpatioTemporalPerformer_Forecaster(
    d_y_context=49,
    d_y_target=49,
    d_x=1,
    d_model=128,
    d_q_k=16,
    d_v=16,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=256,
    time_emb_dim=2,
    activation='relu',
    initial_lr=1e-4,
    loss='mse',
    l2_regul_factor=1e-4,
    verbose=True,
    start_len=16,
    max_seq_len=84,
    normalization_type='batchnorm',
    data_dropout_embedding=0.0,
    dropout_ff=0.0,
    redraw_interval=500,
    use_seasonal_decomp=True,
    use_revin=True,
    use_global_norm=False,
    global_mean=None,
    global_std=None,
    linear_window=10,
    linear_shared_weights=True,
    time_embedding=False,
    indicator_embedding=False,
    decay_factor=0.1,
    warmup_steps=4000,
    num_training_steps=num_trainin_steps,
    autoregressive_training=False,
    scheduled_sampling_k= 1e-10
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cpkt = torch.load('/content/checkpoints/best-checkpoint.ckpt', map_location=device)
state = cpkt['state_dict']

for k in list(state.keys()):
    if 'time_embedding' in k:
        state.pop(k)
        print(k)

model_no_time.load_state_dict(state, strict=False)



saved_model = SpatioTemporalPerformer_Forecaster.load_from_checkpoint('/content/checkpoints/best-checkpoint-v1.ckpt')

import shutil
import os

local_checkpoint_path = '/content/checkpoints/best-checkpoint-v2.ckpt'  # Path to the checkpoint file
local_tb_log_path = '/content/tb_log/test/version_5'

drive_checkpoint_path = '/content/drive/MyDrive/Spatiotemporal_Transformer/simple_timestepping/w_vel/checkpoint'
drive_log_path = '/content/drive/MyDrive/Spatiotemporal_Transformer/simple_timestepping/w_vel/tb_log'

os.makedirs(drive_checkpoint_path, exist_ok=True)
os.makedirs(drive_log_path, exist_ok=True)

# Use shutil.copy2 to copy the checkpoint file
shutil.copy2(local_checkpoint_path, drive_checkpoint_path)
# Use shutil.copytree to copy the tb_log directory
shutil.copytree(local_tb_log_path, drive_log_path, dirs_exist_ok=True)

print("Files copied successfully to Google Drive")

trainer = pl.Trainer()
trainer.validate(saved_model, datamodule=data_module_one)

device = saved_model.device

ctx_x_batch, y_ctx_batch, tgt_x_batch, _ = next(iter(val_loader_one))
ctx_x_batch = ctx_x_batch.to(device)
y_ctx_batch = y_ctx_batch.to(device)
tgt_x_batch = tgt_x_batch.to(device)

hist_vals = y_ctx_batch.clone()

ctx_x, _, _, _ = next(iter(val_loader_one))
print(ctx_x.shape)

device = saved_model.device
saved_model.to(device)
# 1) Put the model into its pure AR branch
saved_model.eval()
#saved_model.autoregressive_training = True
# if you really want absolutely zero teacher-forcing:
#saved_model.scheduled_sampling_k = 1e-6
#model_no_time.time_embedding = False
#model_no_time.indicator_embedding = False

all_preds, all_truths = [], []

# 2) For every sliding‐window in your val_loader_one:
with torch.no_grad():
    for x_ctx, y_ctx, x_tgt, y_tgt in val_loader_one:
        x_ctx = x_ctx.to(device)   # (B, m, p)
        y_ctx = y_ctx.to(device)   # (B, m, D)
        x_tgt = x_tgt.to(device)   # (B, n, p)
        y_tgt = y_tgt.to(device)   # (B, n, D)

        # 3) This single call does an n-step AR rollout internally:
        preds = saved_model.predict(x_ctx, y_ctx, x_tgt)
        # preds.shape == (B, n, D)

        all_preds.append(preds.cpu())
        all_truths.append(y_tgt.cpu())

# 4) Evaluate
all_preds = torch.cat(all_preds, 0)  # (N_windows, n, D)
all_truths = torch.cat(all_truths, 0)
mse = F.mse_loss(all_preds, all_truths)
print("Auto‐regressive MSE:", mse.item())

ctx_x0, y_ctx0, _, _ = next(iter(val_loader_one))
batch_size, m, d_x = ctx_x0.shape
horizon = 1434
x_target_1000 = torch.zeros(batch_size, horizon, d_x, device=device)
model_no_time.eval()
model_no_time.autoregressive_training = True
model_no_time.scheduled_sampling_k    = 1e-6  # ratio ≈ 0

# 3) one‐call multi‐step forecast
with torch.no_grad():
    preds_1000 = model_no_time.predict(
        ctx_x0.to(device),    # (4, 80, 20)
        y_ctx0.to(device),    # (4, 80, 49)
        x_target_1000         # (4,1000,20)
    )

preds = all_preds.numpy()
truths = all_truths.numpy()

preds.shape, truths.shape

preds = preds[:, -1, :]
truths = truths[:, -1, :]
preds.shape, truths.shape

preds_correct = preds.copy()
truths_correct = truths.copy()

preds_wrong = preds.copy()

preds_wrong

truths_wrong = truths

truths

import matplotlib.pyplot as plt

# preds_wrong, preds_correct: both shape (T,) or (T,1)
# truth: shape (T,)
resid_wrong   = (preds_wrong  - truths).flatten()
resid_correct = (preds_correct - truths).flatten()

t = np.arange(len(resid_wrong))
plt.figure(figsize=(10,4))
plt.plot(t, resid_correct, label='correct model err')
plt.plot(t, resid_wrong,   label='wrong model err', alpha=0.7)
plt.legend()
plt.title("Point-wise Residuals Over Time")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(resid_correct, bins=50, alpha=1, label='correct')
plt.hist(resid_wrong,   bins=50, alpha=0.2, label='wrong')
plt.legend()
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(4,4))
plt.scatter(truths.flatten(), preds_correct.flatten(), s=2, label='correct')
plt.scatter(truths.flatten(), preds_wrong.flatten(),   s=1, alpha=0.5, label='wrong')
lims = [truths.min(), truths.max()]
plt.plot(lims, lims, 'k--')
plt.legend()
plt.title("True vs Predicted")
plt.show()

yy = np.zeros((2))
xx = [1, 500]
yy[0]=10

yy[1] = yy[0]*(xx[1]/xx[0])**(-5/3)


plt.plot(xx,yy)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Assume you have the following NumPy arrays:
# truth_full: shape (T, D)
# preds_correct_full: shape (T, D)
# preds_wrong_full:   shape (T, D)
# For this snippet, replace the dummy data with your real arrays.

# Dummy example (remove when using real data)
T, D = 2048, 49
np.random.seed(0)
truth_full = np.sin(2*np.pi*0.02*np.arange(T))[:, None] + 0.05*np.random.randn(T, D)
preds_correct_full = truth_full + 0.01*np.random.randn(T, D)
preds_wrong_full   = truth_full + 0.03*np.random.randn(T, D)

sensor_idx = 8
fs = 1.0  # sampling frequency (samples per time step)

# Extract one sensor's series
x_true = truth_full[:, sensor_idx]
x_corr = preds_correct_full[:, sensor_idx]
x_wrong = preds_wrong_full[:, sensor_idx]

# Compute PSD via Welch's method
f_true, Pxx_true   = welch(x_true, fs=fs, nperseg=512)
f_corr, Pxx_corr   = welch(x_corr, fs=fs, nperseg=512)
f_wrong, Pxx_wrong = welch(x_wrong, fs=fs, nperseg=512)

# Plot
plt.figure(figsize=(6,4))
plt.loglog(f_true,   Pxx_true,   label='Truth',        linewidth=1.5)
plt.loglog(f_corr,   Pxx_corr,   label='Correct Model', linewidth=1.5, alpha=0.8)
plt.loglog(f_wrong,  Pxx_wrong,  label='Wrong Model',   linewidth=1.5, alpha=0.8)

# -5/3 slope reference
f_ref = np.array([f_true[1], f_true[-1]])
C = Pxx_true[1] / (f_true[1]**(-5/3))
plt.loglog(f_ref, C * f_ref**(-5/3), 'k--', label=r'$f^{-5/3}$')

plt.xlabel('Frequency (cycles per time step)', fontsize=12)
plt.ylabel('Power Spectral Density', fontsize=12)
plt.title(f'Sensor {sensor_idx} PSD Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- Convert your PyTorch tensors to NumPy, if needed ---
truths_np      = truths        if isinstance(truths, np.ndarray)        else truths.cpu().numpy()
preds_corr_np  = preds_correct if isinstance(preds_correct, np.ndarray)  else preds_correct.cpu().numpy()
preds_wrong_np = preds_wrong   if isinstance(preds_wrong, np.ndarray)   else preds_wrong.cpu().numpy()

# --- Pick a sensor index and set sampling ---
sensor_idx = 0      # choose 0…(D-1)
dt = 1.0            # one time‐step between samples
fs = 1.0 / dt       # sampling frequency

# --- Extract the time series for that sensor ---
x_true  = truths_np[:,      sensor_idx]
x_wrong = preds_wrong_np[:, sensor_idx]

# --- Estimate PSD via Welch ---
nperseg = min(512, len(x_true))
f, Pxx_true  = welch(x_true,  fs=fs, nperseg=nperseg)
_, Pxx_wrong = welch(x_wrong, fs=fs, nperseg=nperseg)

# --- Compute relative spectral error ---
eps = 1e-12
rel_err = np.abs(Pxx_wrong - Pxx_true) / (Pxx_true + eps)

# --- Plot it ---
plt.figure(figsize=(6,3))
plt.semilogx(f, rel_err, label='|PSD_wrong − PSD_true|/PSD_true')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('Frequency (cycles per time step)')
plt.ylabel('Relative PSD Error')
plt.title(f'Sensor {sensor_idx} Spectral Error')
plt.grid(which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# assuming you’ve already got f, Pxx_true, Pxx_wrong from Welch
rel_err = np.abs(Pxx_wrong - Pxx_true) / (Pxx_true + 1e-12)

plt.figure(figsize=(6,3))
plt.semilogx(f, rel_err, label='Relative Error')
plt.xlabel('Frequency')
plt.ylabel('Relative |ΔPSD|')
plt.title('Spectral Relative Error')
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

x = np.arange(preds.shape[0])
plt.figure(figsize=(12, 6))
plt.plot(x, truths[:, 5], label='truths', color='blue')
plt.plot(x, preds[:, 5], label='preds', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

np.save('/content/drive/MyDrive/Spatiotemporal_Transformer/simple_timestepping/w_vel/results/w_vel_5_10.npy', preds)
np.save('/content/drive/MyDrive/Spatiotemporal_Transformer/simple_timestepping/w_vel/results/w_truth_5_10.npy', truths)

print(x_context)







import torch
import numpy as np

# Load the model from checkpoint
#model = SpatioTemporalPerformer_Forecaster.load_from_checkpoint("/content/checkpoints/best-checkpoint.ckpt")

# Switch to teacher forcing for evaluation (so the model always gets ground truth as input)
model.autoregressive_training = False
model.use_shifted_time_window = False
model.eval()  # Set model to evaluation mode

# Get the validation dataloader from your data module
val_loader = data_module_one.val_dataloader()

all_predictions = []
all_ground_truths = []

# No gradient computation during evaluation
with torch.no_grad():
    for batch in val_loader:
        x_context, y_context, x_target, y_target = batch

        # Get predictions from the model
        preds = model.predict(x_context, y_context, x_target)

        # Append predictions and ground truths (ensure they are moved to CPU and converted to numpy arrays)
        all_predictions.append(preds.cpu().numpy())
        all_ground_truths.append(y_target.cpu().numpy())

# Concatenate all batches into a single array
all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truths = np.concatenate(all_ground_truths, axis=0)

# Calculate numeric metrics
mae = np.mean(np.abs(all_predictions - all_ground_truths))
mse = np.mean((all_predictions - all_ground_truths) ** 2)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

all_predictions.shape, all_ground_truths.shape

import torch
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model from checkpoint.
model = SpatioTemporalPerformer_Forecaster.load_from_checkpoint("/content/checkpoints/best-checkpoint.ckpt")
model = model.to(device)
model.eval()

all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for batch in val_loader_one:
        # Unpack the batch (assumed order: x_context, y_context, x_target, y_target)
        x_context, y_context, x_target, y_target = batch
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Run the forward pass.
        output = model(x_context, y_context, x_target, y_target, output_attention=False)

        # If the output is a tuple, unpack it
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output

        all_predictions.append(predictions.cpu().numpy())
        all_ground_truths.append(y_target.cpu().numpy())

all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truths = np.concatenate(all_ground_truths, axis=0)

mse = np.mean((all_predictions - all_ground_truths) ** 2)
print(f"Validation MSE: {mse:.4f}")

import torch
import numpy as np

# Assume model is already loaded and on the appropriate device.
model.eval()
all_autoregressive_predictions = []
all_ground_truths = []

with torch.no_grad():
    # Loop over validation batches; for each, take the first context only.
    for batch in val_loader_one:
        x_context, y_context, x_target, y_target = batch

        x_context = x_context.to(model.device)
        y_context = y_context.to(model.device)
        x_target = x_target.to(model.device)
        y_target = y_target.to(model.device)

        # We'll use only the first 'context_len' data (80 steps).
        # This assumes your batch data is organized as [batch_size, sequence_length, features].
        # You could simply use the first 80 points as context:
        context_x = x_context[:, :80, :]
        context_y = y_context[:, :80, :]

        # Let the model forecast the next 'target_len' steps (4 steps) in a fully autoregressive mode.
        # Ensure that the model is in autoregressive mode:
        model.autoregressive_training = True

        # Create a dummy x_target slice for the forecast horizon.
        # Depending on your implementation, you might need to provide an x_target or leave it empty.
        forecast = model(context_x, context_y, x_target[:, :4, :], y_target[:, :4, :], output_attention=False)

        # If forecast returns a tuple, unpack it:
        if isinstance(forecast, tuple):
            forecast = forecast[0]

        all_autoregressive_predictions.append(forecast.cpu().numpy())
        all_ground_truths.append(y_target.cpu().numpy())

all_autoregressive_predictions = np.concatenate(all_autoregressive_predictions, axis=0)
all_ground_truths = np.concatenate(all_ground_truths, axis=0)

# Compute evaluation metrics (e.g., MSE)
mse = np.mean((all_autoregressive_predictions - all_ground_truths) ** 2)
print(f"Autoregressive Forecast MSE: {mse:.4f}")

import torch
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model from checkpoint.
model = SpatioTemporalPerformer_Forecaster.load_from_checkpoint("/content/checkpoints/best-checkpoint.ckpt")
model = model.to(device)
model.eval()

all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for batch in val_loader_one:
        # Unpack the batch (assumed order: x_context, y_context, x_target, y_target)
        x_context, y_context, x_target, y_target = batch
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Run the forward pass.
        output = model(x_context, y_context, x_target, y_target, output_attention=False)

        # If output is a tuple, unpack it.
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output

        # If using teacher forcing (i.e., autoregressive_training == False),
        # the output includes the initial seed (start_len). Remove it:
        if not model.autoregressive_training:
            predictions = predictions[:, model.start_len:, :]

        all_predictions.append(predictions.cpu().numpy())
        all_ground_truths.append(y_target.cpu().numpy())

all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truths = np.concatenate(all_ground_truths, axis=0)

mse = np.mean((all_predictions - all_ground_truths) ** 2)
print(f"Validation MSE: {mse:.4f}")

all_predictions.shape

predicted = all_predictions[:, -1, :]
real = all_ground_truths[:, -1, :]
predicted.shape, real.shape

predicted = all_predictions[:, -1, :]
real = all_ground_truths[:, -1, :]
predicted.shape, real.shape

all_ground_truths.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_model.eval()
all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for batch in val_loader_one:
        context_x, context_y, target_x, target_y = batch
        x_context = context_x.to(device)
        #x_context = torch.zeros_like(x_context).to(device)
        y_context = context_y.to(device)
        #y_context = torch.zeros_like(y_context).to(device)
        x_target = target_x.to(device)
        #x_target = torch.zeros_like(x_target).to(device)
        y_target = target_y.to(device)
        output = saved_model.predict(x_context, y_context, x_target)
        all_predictions.append(output.cpu())
        all_ground_truths.append(target_y.cpu())

all_predictions = torch.cat(all_predictions, dim=0)
all_ground_truths = torch.cat(all_ground_truths, dim=0)

import torch
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model from checkpoint.
model = SpatioTemporalPerformer_Forecaster.load_from_checkpoint("/content/checkpoints/best-checkpoint.ckpt")
model = model.to(device)
model.eval()

all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for batch in val_loader_one:
        # Unpack the batch (assumed order: x_context, y_context, x_target, y_target)
        x_context, y_context, x_target, y_target = batch
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Run the forward pass.
        output = model(x_context, y_context, x_target, y_target, output_attention=False)
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output

        # Remove the seed part if using teacher forcing.
        if not model.autoregressive_training:
            predictions = predictions[:, model.start_len:, :]

        all_predictions.append(predictions.cpu().numpy())
        all_ground_truths.append(y_target.cpu().numpy())

all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truths = np.concatenate(all_ground_truths, axis=0)

# Option 1: Compute MSE without shifting:
mse = np.mean((all_predictions - all_ground_truths) ** 2)
print(f"Validation MSE (original): {mse:.4f}")

# Option 2: Try shifting the predictions by one time step to check alignment.
shifted_predictions = np.roll(all_predictions, shift=-1, axis=1)
# For boundary adjustment, you might replace the rolled value at the end with the original last value:
shifted_predictions[:, -1, :] = all_predictions[:, -1, :]

mse_shifted = np.mean((shifted_predictions - all_ground_truths) ** 2)
print(f"Validation MSE (shifted by -1): {mse_shifted:.4f}")

forecasts = np.array(forecasts)
ground_truths = np.array(ground_truths)
forecasts.shape, ground_truths.shape

predicted_results = all_predictions.numpy()
predicted_results = predicted_results[:, -1, :]
predicted_results.shape

real_data = all_ground_truths.numpy()
real_data = real_data[:, -1, :]
real_data.shape

val = data_one[6090:]
val = val[22:]
val.shape

all_predictions.shape

all_ground_truths.shape

real = all_ground_truths[:, -1, :]
predicted = all_predictions[:, -1, :]
real.shape, predicted.shape

real_untrained = all_ground_truths[:, -1, :]
predicted_untrained = all_predictions[:, -1, :]
real.shape, predicted.shape

import matplotlib.pyplot as plt

l = len(predicted)
x = np.arange(l)

plt.figure(figsize=(12, 6))
#plt.plot(x[:], val[:, 1], label='Real Data', color='black')
plt.plot(x[:], predicted_untrained[:, 4], label='Context', color='blue', linewidth=4)
plt.plot(x[:], predicted[:, 4], label='Forecast', color='red', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

train_context = []
train_target = []
for batch in train_loader_one:
    _, context_y, _, target_y = batch
    train_context.append(context_y.cpu().numpy())
    train_target.append(target_y.cpu().numpy())

train_context_one = train_target[:-1]
train_context_one = np.concatenate(train_context_one, axis=0)
train_context_one.shape

np.min(train_context_one)

train_target

