import numpy as np

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


