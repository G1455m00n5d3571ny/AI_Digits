from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# 2. === Подготовка данных ===
# Нормализация входных данных (0-255 -> 0-1)
data_train_norm = data_train.astype('float32') / 255.0
data_test_norm = data_test.astype('float32') / 255.0

print(f'Before norm: [{data_train.min()} - {data_train.max()}]\n'
      f'After norm: [{data_train_norm.min():.2f} - {data_train_norm.max():.2f}]\n')

