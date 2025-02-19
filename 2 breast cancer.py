import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer # набор данных: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA   #  метод главных компонент (готовый)
from sklearn.metrics import r2_score

# 1, 2. Загрузка и подготовка данных
data_breast = load_breast_cancer()
X = data_breast.data       # Входные данные (матрица признаков)
y = data_breast.target     # Выходные данные - Целевая переменная (для сравнения, не используется в PCA)

# Для удобства работы преобразуем в DataFrame и проверим типы данных
df = pd.DataFrame(X, columns=data_breast.feature_names)
#print("Размерность данных:", df.shape)
#print("Типы данных:\n", df.dtypes)
'''
[ Вывод ]

Размерность данных: (569, 30)
Типы данных:
mean radius                float64
mean texture               float64 
mean perimeter             float64 
mean area                  float64 
mean smoothness            float64 
mean compactness           float64 
mean concavity             float64 
mean concave points        float64 
mean symmetry              float64 
mean fractal dimension     float64 
radius error               float64
texture error              float64
perimeter error            float64
area error                 float64
smoothness error           float64
compactness error          float64
concavity error            float64
concave points error       float64
symmetry error             float64
fractal dimension error    float64
worst radius               float64
worst texture              float64
worst perimeter            float64
worst area                 float64
worst smoothness           float64
worst compactness          float64
worst concavity            float64
worst concave points       float64
worst symmetry             float64
worst fractal dimension    float64
dtype: object
'''

# Первичный анализ данных (проверяем наличие пропущенных значений)
#print("Пропущенные значения:\n", df.isna().sum())
'''
[ Вывод ]

Пропущенные значения:
mean radius                0
mean texture               0 
mean perimeter             0 
mean area                  0 
mean smoothness            0 
mean compactness           0 
mean concavity             0 
mean concave points        0 
mean symmetry              0 
mean fractal dimension     0 
radius error               0 
texture error              0 
perimeter error            0 
area error                 0 
smoothness error           0 
compactness error          0 
concavity error            0
concave points error       0
symmetry error             0
fractal dimension error    0
worst radius               0
worst texture              0
worst perimeter            0
worst area                 0
worst smoothness           0
worst compactness          0
worst concavity            0
worst concave points       0
worst symmetry             0
worst fractal dimension    0
dtype: int64
'''

# 3. Реализация метода главных компонент (PCA) - с нуля
# 3.1. Нормализация данных: вычтем среднее для каждого признака
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std # Z-преобразование, по формуле: x' = (x - μ) / σ 

# 3.2. Вычисление ковариационной матрицы (с учётом того, что признаки - столбцы)
cov_matrix = np.cov(X_norm, rowvar=False)
cov_matrix_rounded = np.around(cov_matrix, decimals=1)  # Округляем матрицу до 1 знака после запятой для вывода
#print("Ковариационная матрица имеет размер:", cov_matrix.shape) # Для предварительной проверки
#print("Ковариационная матрица:", cov_matrix_rounded)
'''
[ Вывод ]

Ковариационная матрица имеет размер: (30, 30)
Ковариационная матрица: 
[[ 1.   0.3  1.   1.   0.2  0.5  0.7  0.8  0.1 -0.3  0.7 -0.1  0.7  0.7  -0.2  0.2  0.2  0.4 -0.1 -0.   1.   0.3  1.   0.9  0.1  0.4  0.5  0.7  0.2  0. ]
...
[ 0.   0.1  0.1  0.   0.5  0.7  0.5  0.4  0.4  0.8  0.  -0.   0.1  0.  0.1  0.6  0.4  0.3  0.1  0.6  0.1  0.2  0.1  0.1  0.6  0.8  0.7  0.5  0.5  1. ]]
'''

# 3.3. Диагонализация матрицы: получение собственных значений и векторов
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# 3.4. Сортировка собственных значений и векторов по убыванию
sorted_indices = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[sorted_indices]
eig_vecs_sorted = eig_vecs[:, sorted_indices]
#print("Первые 15 собственных значений:", eig_vals_sorted[:15])
#print("Сортировка:", sorted_indices)
'''
[ Вывод ]

Первые 15 собственных значений: 
[   13.30499079  5.7013746   2.82291016  1.98412752  1.65163324  
    1.20948224   0.67640888  0.47745625  0.41762878  0.35131087  
    0.29443315   0.26162116  0.24178242  0.15728615  0.0943007 ]
Сортировка:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 28 29 27 26 25
 24 23 22 21 20 19]
'''
# 3.5. Отбор K главных компонент
K = 15  # Судя по выводу выше первых 15 собственных значений, хватило бы и K = 6. По методу локтя - K = 3.
eig_vecs_K = eig_vecs_sorted[:, :K]

# 3.6. Проекция нормализованных данных на выбранные K главных компонент
X_reduced = X_norm.dot(eig_vecs_K)
#print("Размерность понижения:", X_reduced.shape) # 3.7. Возврат данных сниженной размерности
'''
[ Вывод ]

Размерность понижения: (569, 15)
'''

# 4. Использование встроенной реализации PCA из scikit-learn для сравнения
pca = PCA(n_components=K)
X_pca = pca.fit_transform(X)
#print("Объяснённая дисперсия (manual):", eig_vals_sorted[:K] / np.sum(eig_vals_sorted))
#print("Объяснённая дисперсия (sklearn):", pca.explained_variance_ratio_)
'''
[ Вывод ]

Объяснённая дисперсия (manual): 
[0.44272026 0.18971182 0.09393163 0.06602135 0.05495768 0.04024522
 0.02250734 0.01588724 0.01389649 0.01168978 0.00979719 0.00870538
 0.00804525 0.00523366 0.00313783]
Объяснённая дисперсия (sklearn): 
[9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
 8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
 3.44135279e-07 1.86018721e-07 6.99473206e-08 1.65908880e-08
 6.99641650e-09 4.78318306e-09 2.93549214e-09]
'''

# 5. Визуализация модели
# Построим диаграмму рассеяния для первых двух главных компонент (manual)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("Диаграмма рассеяния: PCA (собственная реализация)")
plt.show()
'''
[ Вывод ]

Видно, при каких значениях первой и второй главных компонент определяется рак. 
Данные можно хорошо разбить по двум классам (серые и зелёные точки).
'''

# 6. Определение оптимального числа главных компонент (методом локтя)
explained_variances = []    # Список для накопления объяснённой дисперсии
num_features = X.shape[1]   # Количество признаков (столбцов) в матрице данных X
for k in range(1, num_features + 1):
    pca_k = PCA(n_components=k) # Извлекает k главных компонент
    pca_k.fit(X)    # Модель обучается на данных X, вычисляются главные компоненты
    explained_variances.append(np.sum(pca_k.explained_variance_ratio_)) # Массив, где элементы – 
    # – доли от общей дисперсии, объяснённой соответствующей компонентой. Суммируем.
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_features + 1), explained_variances, marker='o')
# plt.gca().invert_yaxis() # отзеркаливаем график по горизонтали для более привычного вида
plt.xlabel("Количество главных компонент")
plt.ylabel("Суммарное объяснённое соотношение дисперсии")
plt.title("Метод локтя для определения оптимального числа компонент")
plt.grid(True)
plt.show()
'''
[ Вывод ]

k = 2 или 3
'''
