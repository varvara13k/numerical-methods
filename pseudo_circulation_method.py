import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy as np
import math

# Метод псевдообращения
def pseudo_circulation_method(W, p):
    pseudo_restored_image = np.linalg.pinv(W.T) @ p # решение матричного изображения
    N = int(math.sqrt(W.shape[0]))  # сторона восстановленного изображения
    return pseudo_restored_image.reshape(N, N)

# расчет квадратической ошибки
def squared_error(original_image, restored_image):
    return round(np.mean((restored_image - original_image).T @ (restored_image - original_image)), 3)

# пример использования

original_image = np.zeros((20, 20))     # создание исходного изображения
square_top_left = (5, 5)    # рисуем квадрат, задавая координаты верхнего левого и нижнего правого углов
square_bottom_right = (15, 15)
original_image[square_top_left[0]:square_bottom_right[0] + 1,
square_top_left[1]:square_bottom_right[1] + 1] = 255


original_image /= 255   # масштабирование

N = original_image.shape[0]     # оригинал и восстановленное изображение имеют размеры N*N
K = N - 2   # вектор-отображение имеет размер K*1

W = np.random.normal(0, 0.01, (N**2 , K **2))   # матрица отображения
n = np.random.normal(0, 0.01, W.shape[1])   # создание вектора шума
p = W.T @ original_image.flatten() + n      # вектор отображения
p /= np.max(p)

# получение восстановленного изображения

pseudo_restored_image = pseudo_circulation_method(W, p)

# визуализация исходного изображения

plt.subplot(131)
plt.imshow(original_image, cmap='gray')
plt.title('Исходное изображение')

# отобразить размытое изображение с шумом

plt.subplot(132)
plt.imshow(p.reshape(K, K), cmap='gray')
plt.title('Размытое изображение с шумом')

# отобразить восстановленные изображения
plt.title('Результат')
plt.subplot(133)
plt.imshow(pseudo_restored_image, cmap='gray')
plt.title('Метод псевдообращения')

plt.show()

# вычисление квадратической ошибки

pseudo_squared_error = squared_error(original_image, pseudo_restored_image)
print("СКО для метода псевдообращения:", pseudo_squared_error)

