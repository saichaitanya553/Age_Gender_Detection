import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _relu(x):
    return np.maximum(x, 0.0)


def _max_pool_2x2(x):
    # Keras MaxPooling2D(pool_size=2, strides=2, padding='valid')
    h = (x.shape[1] // 2) * 2
    w = (x.shape[2] // 2) * 2
    x = x[:, :h, :w, :]
    return x.reshape(x.shape[0], h // 2, 2, w // 2, 2, x.shape[3]).max(axis=(2, 4))


def _conv3x3_valid(x, kernel, bias):
    # Keras Conv2D with channels_last, stride 1, valid padding, groups=1.
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    # windows: batch, height, width, channels, 3, 3
    y = np.einsum("bhwcxy,xyco->bhwo", windows, kernel, optimize=True)
    return y + bias


class AgeCNN:
    def __init__(self, path=None):
        path = path or os.path.join(BASE_DIR, "age_weights.npz")
        self.w = np.load(path)

    def __call__(self, x):
        x = _relu(_conv3x3_valid(x, self.w["conv1_kernel"], self.w["conv1_bias"]))
        x = _max_pool_2x2(x)
        x = _relu(_conv3x3_valid(x, self.w["conv2_kernel"], self.w["conv2_bias"]))
        x = _max_pool_2x2(x)
        x = _relu(_conv3x3_valid(x, self.w["conv3_kernel"], self.w["conv3_bias"]))
        x = _max_pool_2x2(x)
        x = x.reshape(x.shape[0], -1)
        x = _relu(x @ self.w["dense_kernel"] + self.w["dense_bias"])
        return x @ self.w["out_kernel"] + self.w["out_bias"]


class GenderCNN:
    def __init__(self, path=None):
        path = path or os.path.join(BASE_DIR, "gender_weights.npz")
        self.w = np.load(path)

    def __call__(self, x):
        x = _relu(_conv3x3_valid(x, self.w["conv1_kernel"], self.w["conv1_bias"]))
        x = _max_pool_2x2(x)
        x = _relu(_conv3x3_valid(x, self.w["conv2_kernel"], self.w["conv2_bias"]))
        x = _max_pool_2x2(x)
        x = x.reshape(x.shape[0], -1)
        x = _relu(x @ self.w["dense_kernel"] + self.w["dense_bias"])
        logits = x @ self.w["out_kernel"] + self.w["out_bias"]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
