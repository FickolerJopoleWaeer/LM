import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris  # набор данных: Ирисы, https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering # AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1.
'''
классификация: 
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
кластеризация: 
https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html 
https://habr.com/ru/articles/798331/
'''
# -------- Ирисы --------

# 2. Загрузка набора данных "Ирисы Фишера"
# Загрузка датасета Iris из scikit-learn
iris = load_iris()
'''
[ Информация по дата сету ]

Классы	3
Количество образцов по классу	50
Всего образцов	150
Размерность	4
Переменные:
sepal length - длина чашелистика
sepal width - ширина чашелистика
petal length - длина лепестка
petal width - ширина лепестка
'''
X_iris = iris.data       # Признаки (размерность: 4)
y_iris = iris.target     # Целевая переменная (3 класса)
#print("Размерность данных:", X_iris.shape)
#print("Переменные:", iris.feature_names)
'''
[ Вывод ]

Размерность данных: (150, 4)
Переменные: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
'''


# 3. Классификация с использованием метода k-ближайших соседей (kNN)
# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=50)    # 30% - тестовая выборка, 70% - обучающая. 
# Создадим и обучим классификатор kNN (с 5 ближайшими соседями)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("Точность метода k ближайших соседей (kNN):", acc_knn)
# Для визуализации решающих границ используем Метод главных компонент (PCA) для уменьшения размерности до 2
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)
# Обучим kNN на данных, преобразованных с помощью PCA:
X_train_pca, X_test_pca, _, _ = train_test_split(X_iris_pca, y_iris, test_size=0.3, random_state=50)
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
# Сетка для визуализации:
x_min, x_max = X_iris_pca[:, 0].min() - 1, X_iris_pca[:, 0].max() + 1
y_min, y_max = X_iris_pca[:, 1].min() - 1, X_iris_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.title(f"k ближайших соседей (Ирисы) | Точность: {round(acc_knn, 3)}")
#plt.show()
'''
[ Вывод ]
Точность метода k ближайших соседей (kNN): 0.933
Лит-ра: https://habr.com/ru/articles/149693/
Для повышения надёжности классификации объект относится к тому классу, 
которому принадлежит большинство из его соседей — ближайших к нему объектов обучающей выборки 
'''

# 4. Классификация с использованием алгоритма "Случайный лес"
rf = RandomForestClassifier(n_estimators=100, random_state=50)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Точность случайного леса:", acc_rf)
# Визуализируем случайный лес (Random Forest) на PCA-преобразованных данных
rf_pca = RandomForestClassifier(n_estimators=100, random_state=50)
rf_pca.fit(X_train_pca, y_train)
Z_rf = rf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rf = Z_rf.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_rf, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.title(f"Cлучайный лес (Ирисы) | Точность: {round(acc_rf, 3)}")
#plt.show()
'''
[ Вывод ]
Точность случайного леса: 0.956
Случайный лес (Random Forest) обычно превосходит отдельное дерево решений за счёт уменьшения переобучения 
– каждое дерево обучается на случайном подмножестве признаков, что обеспечивает лучшую обобщаемость модели.
'''

# 5. Классификация с использованием метода опорных векторов (SVM)
svm = SVC(kernel='linear', random_state=50)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Точность метода опорных векторов:", acc_svm)
# Визуализируем SVM на PCA-преобразованных данных
svm_pca = SVC(kernel='linear', random_state=50)
svm_pca.fit(X_train_pca, y_train)
Z_svm = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.title(f"Метод опорных векторов (Ирисы) | Точность: {round(acc_svm, 3)}")
#plt.show()
'''
[ Вывод ]
Точность метода опорных векторов: 0.956
Основная идея метода — перевод исходных векторов в пространство более высокой размерности 
и поиск разделяющей гиперплоскости с наибольшим зазором в этом пространстве. 
Две параллельных гиперплоскости строятся по обеим сторонам гиперплоскости, разделяющей классы.
'''

# 6. Кластеризация с использованием алгоритма k-средних
kmeans = KMeans(n_clusters=3, random_state=50) # задаём 3 кластера, т.к. знаем, соклько их изначально
clusters = kmeans.fit_predict(X_iris)
plt.figure(figsize=(8, 6))
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')  # используем данные, преобразованных с помощью метода главных компонент - PCA
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.title("k-средних, кластеризация (Ирисы)")
#plt.show()
'''
Основная идея заключается в том, что на каждой итерации перевычисляется центр масс для каждого кластера, 
полученного на предыдущем шаге, затем векторы разбиваются на кластеры вновь в соответствии с тем, 
какой из новых центров оказался ближе по выбранной метрике.
'''

# 7. Иерархическая кластеризация методом Уорда
# Производим агломеративную кластеризацию и строим дендрограмму
linked = linkage(X_iris, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=iris.target, leaf_rotation=90)
plt.title("дендрограмма кластеризации, кластеризация (метод Уорда, Ирисы)")
plt.xlabel("Выборочный индекс")
plt.ylabel("Расстояние")
#plt.show()
'''
Иерархическая кластеризация - вначале рассматривается каждое наблюдение (переменная) 
как отдельный кластер, а затем последовательно объединяются кластеры, пока не останется только один. - *Агломеративный (восходящий)

Метод Уорда: расстояние между кластерами равно приросту суммы квадратов 
расстояний от точек до центроидов кластеров при объединении этих кластеров
'''

# 8. Спектральная кластеризация
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=50)
spectral_labels = spectral.fit_predict(X_iris)
plt.figure(figsize=(8, 6))
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=spectral_labels, cmap='coolwarm', edgecolor='k')
plt.xlabel("Компонента 1")
plt.ylabel("Компонента 2")
plt.title("Спектральная кластеризация (Ирисы)")
plt.show()
'''
Матрица сходства подаётся в качестве входа и состоит из количественных оценок относительной схожести каждой пары точек в данных.
Основная идея заключается в преобразовании матрицы сходства графа в лаплассиан для получения его собственных векторов, 
которые в дальнейшем используются для проекции данных в новое пространство более низкой размерности для лучшей разделимости, 
где затем применяется другой метод кластеризации, например, такой как K-средних.
'''