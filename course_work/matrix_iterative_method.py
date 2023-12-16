import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy as np

def iterative_computational_procedure(W, p, F, max_iterations=100):  # матрично-итерационный метод
    background = [] # общий список для хранения индексов фоновых точек

    for k in range(max_iterations):
        f = np.linalg.pinv(W.T) @ p # восстановление изображения на очередном шаге
        less_than_threshold = []    # текущий список для хранения индексов фоновых точек
        current_col_cnt = 0 # счетчик столбцов с замененными значениями

        # поиск фоновых точек
        for i in range(f.shape[0]):
            if f[i] < F:
                less_than_threshold.append(i)
                current_col_cnt += 1
        if current_col_cnt == 0: # если фоновых значений больше нет, то завершаем цикл
            break
        background += less_than_threshold
        f[less_than_threshold] = F  # если i-я координата по значению ниже порога, то заменить на фоновые значения


        W = np.delete(W, less_than_threshold, axis=0) # исключение из матрицы W столбцов с замененными значениями
        f = np.delete(f, less_than_threshold, axis=0) # исключение из f замененных координат

        p = W.T @ f # пересчет вектора-отображения
        p /= np.max(p)

    # дополнение восстановленного изображения фоновыми значениями до исходных размеров
    result = np.array([F] * original_image.flatten().shape[0])
    for i in range(len(result)):
        if i not in background:
            result[i] = f[0]
            f = np.delete(f, 0)
    return result.reshape(original_image.shape)

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

# получение восстановленного изображения каждым из 3-х методов

iterative_restored_image = iterative_computational_procedure(W, p, 0)


# визуализация исходного изображения

plt.subplot(131)
plt.imshow(original_image, cmap='gray')
plt.title('Исходное изображение')

# отобразить размытое изображение с шумом

plt.subplot(132)
plt.imshow(p.reshape(K, K), cmap='gray')
plt.title('Размытое изображение с шумом')

# отобразить восстановленное изображение
plt.subplot(133)
plt.imshow(iterative_restored_image, cmap='gray')
plt.title('Матрично-итерационный метод')
plt.show()


# вычисление квадратической ошибки

iterative_squared_error = squared_error(original_image, iterative_restored_image)
print("СКО для матрично-итерационного метода:", iterative_squared_error)
