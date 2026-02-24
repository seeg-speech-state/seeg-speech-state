from loguru import logger
from scipy import signal
from enum import Enum
import numpy as np

Q = 30.0  # Quality factor
sfreq = 2e3  # Sample frequency (Hz)
stop_freq = [50, 100]  # Frequency to be removed from signal (Hz)


class SignalFilter(str, Enum):
    DEFAULT = "default"
    DEFAULT_V2 = "default_v2"
    HIGH_GAMMA = "high_gamma"

    def abbr(self):
        if self == SignalFilter.DEFAULT:
            return "def"
        elif self == SignalFilter.HIGH_GAMMA:
            return "hg"
        elif self == SignalFilter.DEFAULT_V2:
            return "def2"
        else:
            raise ValueError("invalid SignalFilter value {}", self)


def get_default_sig_filter() -> tuple[list, list]:
    b = []
    a = []

    # low pass filter
    cheby1_b, cheby1_a = signal.cheby1(5, 0.01, 200, btype='low', analog=False, output='ba', fs=sfreq) # type: ignore
    b.append(cheby1_b)
    a.append(cheby1_a)

    # High pass filter 
    # cheby1_b, cheby1_a = signal.cheby1(5, 0.01, 0.5, btype='highpass', analog=False, output='ba', fs=sfreq)
    # notch_a.append(cheby1_a)
    # notch_b.append(cheby1_b)

    for freq in stop_freq:
        notch_b, notch_a = signal.iirnotch(w0=freq, Q=Q, fs=sfreq)
        b.append(notch_b)
        a.append(notch_a)
    
    return b, a


def get_default_v2_sig_filter() -> tuple[list, list]:
    b = []
    a = []

    # low pass filter
    cheby1_b, cheby1_a = signal.cheby1(5, 0.01, [2,200], btype='band', analog=False, output='ba', fs=sfreq) # type: ignore
    b.append(cheby1_b)
    a.append(cheby1_a)

    # High pass filter 
    # cheby1_b, cheby1_a = signal.cheby1(5, 0.01, 0.5, btype='highpass', analog=False, output='ba', fs=sfreq)
    # notch_a.append(cheby1_a)
    # notch_b.append(cheby1_b)

    for freq in stop_freq:
        notch_b, notch_a = signal.iirnotch(w0=freq, Q=Q, fs=sfreq)
        b.append(notch_b)
        a.append(notch_a)
    
    return b, a


def get_hg_sig_filter() -> tuple[list, list]:
    b = []
    a = []
    # band pass filter
    cheby1_b, cheby1_a = signal.cheby1(5, 0.01, [75,150], btype='band', analog=False, output='ba', fs=sfreq) # type: ignore
    b.append(cheby1_b)
    a.append(cheby1_a)

    for freq in stop_freq:
        notch_b, notch_a = signal.iirnotch(w0=freq, Q=Q, fs=sfreq)
        b.append(notch_b)
        a.append(notch_a)

    return b, a


def apply_sig_filter(b: list, a: list, data: np.ndarray):
    filtered_data = data
    for _b, _a in zip(b, a):
        filtered_data = signal.filtfilt(_b, _a, filtered_data, axis=-1)
    return filtered_data


def resample(data: np.ndarray, resample_factor: float):
    raw_len = data.shape[-1]
    return signal.resample(data, int(raw_len * resample_factor), axis=-1)


def ema_filter(data: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    if data.ndim == 3:
        N, C, T = data.shape
        X = data.transpose((0, 2, 1)).reshape(-1, C) # (N, C, T) -> (N, T, C) -> (N * T, C)
        scaler = StandardScaler().fit(X)
        logger.debug("{}", scaler)
        logger.debug("len(mean)={}, len(scale)={}, N={}, C={}, T={}", len(scaler.mean_), len(scaler.scale_), N, C, T) # type: ignore
        X_scaled = scaler.transform(X)
        X_out = X_scaled.reshape(N, T, C).transpose((0, 2, 1))
        assert X_out.shape == (N, C, T)
        return X_out
    elif data.ndim == 2:
        C, T = data.shape
        X = data.T # (C, T) -> (T, C)
        scaler = StandardScaler().fit(X)
        logger.debug("{}", scaler)
        logger.debug("len(mean)={}, len(scale)={}, C={}, T={}", len(scaler.mean_), len(scaler.scale_), C, T) # type: ignore
        X_scaled = scaler.transform(X)
        X_out = X_scaled.T
        assert X_out.shape == (C, T)
        return X_out
    else:
        raise ValueError(f"data with shape={data.shape}, ndim={data.ndim} is unsupported")


# usage:
# for seeg_file in tqdm(seeg_files, desc=f'Processing {seeg_folder}', total=len(seeg_files)):
#         # if os.path.exists(seeg_file.replace('seg_seeg', 'seg_tf')): continue
#         try:
#             seeg_data = np.load(seeg_file)
#             # Apply notch filter
#             filtered_data = seeg_data
#             for a,b in zip(notch_a, notch_b):
#                 filtered_data = signal.filtfilt(b, a, filtered_data)