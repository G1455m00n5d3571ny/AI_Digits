from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import to_categorical

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

# One-hot encoding
target_train_cat = to_categorical(target_train, num_classes = 10)
target_test_cat = to_categorical(target_test, num_classes = 10)

print(f'\nBefore one-hot: {target_train.shape}\n'
      f'After one-hot: {target_train_cat.shape}\n'
      f'Example target[0]: {target_train[0]} -> {target_train_cat[0]}\n')

# Завершение препроцессинга
print(f'Finished preprocessing\n'
      f'data_train_flat: {data_train_flat.shape}\n'
      f'data_test_flat: {data_test_flat.shape}\n'
      f'target_train_cat: {target_train_cat.shape}\n'
      f'target_test_cat: {target_test_cat.shape}\n')
