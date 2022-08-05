import numpy as np
from scipy import signal

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def convolve2d(z, f):
  return signal.convolve2d(z, f, mode='same')