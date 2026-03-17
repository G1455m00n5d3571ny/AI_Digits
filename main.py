from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# 2. === Подготовка данных ===
# Нормализация входных данных (0-255 -> 0-1)
data_train_norm = data_train.astype('float32') / 255.0
data_test_norm = data_test.astype('float32') / 255.0

print(f'Before norm: [{data_train.min()} - {data_train.max()}]\n'
      f'After norm: [{data_train_norm.min():.2f} - {data_train_norm.max():.2f}]\n')

# Reshape формы для Dense
data_train_flat = data_train_norm.reshape(-1, 28 * 28)
data_test_flat = data_test_norm.reshape(-1, 28 * 28)

print(f'Before reshape:\n'
      f'data_train_norm: {data_train_norm.shape}\n'
      f'Single digit: {data_train_norm[0].shape}\n')

print(f'After reshape:\n'
      f'data_train_flat: {data_train_flat.shape}\n'
      f'Single digit: {data_train_flat[0].shape}\n')

print(f'Check: {28} * {28} = {28 * 28}')