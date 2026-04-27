import numpy as np


def _average_by_interval(arr: np.ndarray, interval: float = 1.0) -> np.ndarray:
    """Bin a (time, voltage) array into fixed-width intervals and return bin averages."""
    valid = arr[np.all(np.isfinite(arr), axis=1)]
    time, voltage = valid[:, 0], valid[:, 1]

    sort_idx = np.argsort(time)
    time, voltage = time[sort_idx], voltage[sort_idx]

    bins = np.arange(0, time[-1] + interval, interval)[:-1]
    result = []
    for t_start in bins[:-1]:
        t_end = t_start + interval
        mask = (time >= t_start) & (time < t_end)
        if np.any(mask):
            result.append([(t_start + t_end) / 2, np.mean(voltage[mask])])
    return np.array(result)


def decay_rate(arr: np.ndarray) -> float:
    """Compute voltage decay rate in µV/h (≡ mV/kh) using first and last 1-hour windows."""
    x, y = arr[:, 0], arr[:, 1]
    end_t = x.max()
    mask_start = (x > x[0]) & (x <= x[0] + 1)
    mask_end = (x > end_t - 1.0) & (x <= end_t)
    delta_y = y[mask_end].mean() - y[mask_start].mean()
    delta_x = end_t - 0.5 - 1.5
    return delta_y / delta_x * 1e6


def lifetime(arr: np.ndarray, ratio: float = 0.05) -> float | None:
    """Return the time at which voltage degrades by `ratio` fraction, or None if not reached."""
    binned = arr if (arr[1, 0] - arr[0, 0]) >= 1.0 else _average_by_interval(arr, 1.0)
    time = binned[:, 0]
    voltage = binned[:, 1]

    sort_idx = np.argsort(time)
    time, voltage = time[sort_idx], voltage[sort_idx]

    begin_mask = (time > time[0]) & (time <= time[0] + 1)
    begin_v = voltage[begin_mask].mean()
    threshold = begin_v * (1.0 + ratio)

    end_t = None
    is_above = False
    for t, v in zip(time, voltage):
        if v >= threshold and not is_above:
            is_above = True
            end_t = t
        elif v < threshold:
            is_above = False

    return end_t


def compute_errors(true_value: float, pred_value: float) -> tuple[float, float]:
    """Return (absolute error, absolute percentage error)."""
    ae = abs(true_value - pred_value)
    ape = float('inf') if true_value == 0 else abs((true_value - pred_value) / true_value) * 100
    return ae, ape
