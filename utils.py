import numpy as np
from scipy import signal

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def convolve2d(z, f):
  return signal.convolve2d(z, f, mode='same')

def to_float(img):
    if len(img.shape) >= 3:
        return np.array([to_float(i) for i in img])
    else:
        return (img / 255.).astype(np.float32)

def to8b(img):
    if len(img.shape) >= 3:
        return np.array([to8b(i) for i in img])
    else:
        return (255 * np.clip(np.nan_to_num(img), 0, 1)).astype(np.uint8)

