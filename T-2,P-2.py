import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# set seed
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# load data
(x_train_m, y_train_m), (x_test_m, y_test_m) = tf.keras.datasets.mnist.load_data()
(x_train_c, y_train_c), (x_test_c, y_test_c) = tf.keras.datasets.cifar10.load_data()

# (a) shapes
print("MNIST train:", x_train_m.shape)
print("MNIST test:", x_test_m.shape)
print("CIFAR10 train:", x_train_c.shape)
print("CIFAR10 test:", x_test_c.shape)

# (b) dtype and range
print("\nMNIST dtype:", x_train_m.dtype)
print("MNIST min:", x_train_m.min(), "max:", x_train_m.max())

print("\nCIFAR10 dtype:", x_train_c.dtype)
print("CIFAR10 min:", x_train_c.min(), "max:", x_train_c.max())

# (c) class count
u, c = np.unique(y_train_m, return_counts=True)
print("\nMNIST class count:")
for i, j in zip(u, c):
    print(i, ":", j)

# image grid
names = ['airplane','automobile','bird','cat','deer',
         'dog','frog','horse','ship','truck']

plt.figure(figsize=(15,5))

for i in range(10):
    k = np.random.randint(0, len(x_train_m))
    plt.subplot(2,10,i+1)
    plt.imshow(x_train_m[k], cmap='gray')
    plt.title(str(y_train_m[k]))
    plt.axis('off')

for i in range(10):
    k = np.random.randint(0, len(x_train_c))
    plt.subplot(2,10,i+11)
    plt.imshow(x_train_c[k])
    plt.title(names[y_train_c[k][0]])
    plt.axis('off')

plt.tight_layout()
plt.savefig("dataset_samples.png")
print("\nSaved dataset_samples.png")
plt.show()