"""Export the trained Keras .h5 weights into compact NumPy .npz files.

This script is for model maintenance/export only; it is NOT needed at runtime.
It requires h5py. The resulting .npz files are consumed by numpy_cnn.py.
"""
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EXPORTS = {
    "age_model.h5": [
        ("conv1_kernel", "model_weights/conv2d/sequential/conv2d/kernel"),
        ("conv1_bias", "model_weights/conv2d/sequential/conv2d/bias"),
        ("conv2_kernel", "model_weights/conv2d_1/sequential/conv2d_1/kernel"),
        ("conv2_bias", "model_weights/conv2d_1/sequential/conv2d_1/bias"),
        ("conv3_kernel", "model_weights/conv2d_2/sequential/conv2d_2/kernel"),
        ("conv3_bias", "model_weights/conv2d_2/sequential/conv2d_2/bias"),
        ("dense_kernel", "model_weights/dense/sequential/dense/kernel"),
        ("dense_bias", "model_weights/dense/sequential/dense/bias"),
        ("out_kernel", "model_weights/dense_1/sequential/dense_1/kernel"),
        ("out_bias", "model_weights/dense_1/sequential/dense_1/bias"),
    ],
    "gender_model.h5": [
        ("conv1_kernel", "model_weights/conv2d_3/sequential_1/conv2d_3/kernel"),
        ("conv1_bias", "model_weights/conv2d_3/sequential_1/conv2d_3/bias"),
        ("conv2_kernel", "model_weights/conv2d_4/sequential_1/conv2d_4/kernel"),
        ("conv2_bias", "model_weights/conv2d_4/sequential_1/conv2d_4/bias"),
        ("dense_kernel", "model_weights/dense_2/sequential_1/dense_2/kernel"),
        ("dense_bias", "model_weights/dense_2/sequential_1/dense_2/bias"),
        ("out_kernel", "model_weights/dense_3/sequential_1/dense_3/kernel"),
        ("out_bias", "model_weights/dense_3/sequential_1/dense_3/bias"),
    ],
}

for model_file, paths in EXPORTS.items():
    source = ROOT / "original_models" / model_file
    output = ROOT / ("age_weights.npz" if model_file.startswith("age_") else "gender_weights.npz")
    with h5py.File(source, "r") as h5:
        data = {name: np.asarray(h5[path], dtype=np.float32) for name, path in paths}
    np.savez_compressed(output, **data)
    print(f"Exported {output}")
