import scipy.signal
import torch


def discount(x, gamma):
    """Compute discounted sum along the 0-th dimension of the `x` ndarray
    Return an ndarray `y` with the same shape as x, satisfying:
        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1

    Args:
        x (np.ndarray): 2-D array of floats, time x features
        gamma (float): Discount factor
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

