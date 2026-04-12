

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

print("Seeds set to 42")

print("\nPackage Versions:")
print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", plt.__version__)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\nGPU is available")
    print("Number of GPUs:", len(gpus))
else:
    print("\nNo GPU found")

    # CPU is slower because it has fewer cores and is not optimized for parallel operations.
    # GPU has many cores and can do matrix operations much faster.
    # On GPU machine, I would use GPU acceleration to train models faster.