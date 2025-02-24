import pandas as pd # pip install pandas | Используем pandas DataFrame, чтобы работать с таблицей данных.


# Загрузка данных из CSV
data = pd.read_csv("weatherHistory.csv")
# Просмотр первых строк и типов данных
'''
[ результаты проверки данных ]

print(data.head())
                  Formatted Date        Summary Precip Type  Temperature (C)  ...  Visibility (km)  Loud Cover  Pressure (millibars)                      Daily Summary
0  2006-04-01 00:00:00.000 +0200  Partly Cloudy        rain         9.472222  ...          15.8263         0.0               1015.13  Partly cloudy throughout the day.
1  2006-04-01 01:00:00.000 +0200  Partly Cloudy        rain         9.355556  ...          15.8263         0.0               1015.63  Partly cloudy throughout the day.
2  2006-04-01 02:00:00.000 +0200  Mostly Cloudy        rain         9.377778  ...          14.9569         0.0               1015.94  Partly cloudy throughout the day.
3  2006-04-01 03:00:00.000 +0200  Partly Cloudy        rain         8.288889  ...          15.8263         0.0               1016.41  Partly cloudy throughout the day.
4  2006-04-01 04:00:00.000 +0200  Mostly Cloudy        rain         8.755556  ...          15.8263         0.0               1016.51  Partly cloudy throughout the day.

print(data.dtypes)  # проверим тип данных, чтобы убедиться, что все числовые данные хранятся корректно
[5 rows x 12 columns]
Formatted Date               object
Summary                      object
Precip Type                  object
Temperature (C)             float64
Apparent Temperature (C)    float64
Humidity                    float64
Wind Speed (km/h)           float64
Wind Bearing (degrees)      float64
Visibility (km)             float64
Loud Cover                  float64
Pressure (millibars)        float64
Daily Summary                object
dtype: object
# Можно использовать data["Summary"] = data["Summary"].astype("category") для экономии памяти

print("Размерность:", data.shape) # узнаем размерность таблицы (количество строк и столбцов)
Размерность: (96453, 12)

print("Пропущенные значения:\n", data.isna().sum()) # проверим кол-во пропущенных значений
Пропущенные значения:
Formatted Date                0
Summary                       0 
Precip Type                 517     # содержат значения null, возможно, в некоторые дни просто не было осадков. Можно использовать data["Precip Type"] = data["Precip Type"].fillna("None") для замены
Temperature (C)               0 
Apparent Temperature (C)      0 
Humidity                      0 
Wind Speed (km/h)             0 
Wind Bearing (degrees)        0 
Visibility (km)               0 
Loud Cover                    0 
Pressure (millibars)          0 
Daily Summary                 0 
dtype: int64
'''

# pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#--------          Простая модель          --------

# Выбираем нужные столбцы
X = data[["Humidity"]]      # влажность 
y = data["Apparent Temperature (C)"]    # ощущаемая температура

# Разбиваем данные на тренировочную и тестовую выборки (train_test_split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Обучаем модель линейной регрессии, используя LinearRegression из Scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказываем и оцениваем качество модели
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Коэффициент детерминации R² (только влажность):", r2)
'''
[ результат ]
Коэффициент детерминации R² (только влажность): 0.3587532329296208
'''

# Построим диаграмму рассеяния (scatter plot) для визуального сравнения исходных данных и линии регрессии
import matplotlib.pyplot as plt # pip install matplotlib
import seaborn as sns   # pip install seaborn

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X["Humidity"], y=y, label="Данные")
# Линия регрессии: предсказанные значения для всех данных
sns.lineplot(x=X["Humidity"], y=model.predict(X), color="red", label="Линия регрессии")
plt.xlabel("Влажность")
plt.ylabel("Ощущаемая температура (°C)")
plt.title("Диаграмма рассеяния и линия регрессии (влажность → температура)")
plt.legend()
plt.show()


#--------          Усложнённая модель: добавление скорости ветра          --------

# Выбираем два признака: влажность и скорость ветра
X2 = data[["Humidity", "Wind Speed (km/h)"]]

# Разбиваем данные
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=50)

# Обучаем новую модель
model2 = LinearRegression()
model2.fit(X2_train, y_train)

# Предсказываем температуру и оцениваем надёжность модели
y2_pred = model2.predict(X2_test)
r2_2 = r2_score(y_test, y2_pred)
print("Коэффициент детерминации R² (влажность и скорость ветра):", r2_2)
'''
[ результат ]
Коэффициент детерминации R² (влажность и скорость ветра): 0.3978063026358415
'''

# Для двух признаков построим 3D-график и цветовую карту
from mpl_toolkits.mplot3d import Axes3D as ax

# 3D-график
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X2["Humidity"], X2["Wind Speed (km/h)"], y, c='blue', marker='o')
ax.set_xlabel("Влажность")
ax.set_ylabel("Скорость ветра (км/ч)")
ax.set_zlabel("Ощущаемая температура (°C)")
plt.title("3D-диаграмма: влажность, скорость ветра и температура")
plt.show()

# цветовая карта
# по оси X отображается влажность, по оси Y — скорость ветра, а цвет точек соответствует ощутимой температуре:
plt.figure(figsize=(10, 7))
# Создаём scatter plot, где параметр 'c' принимает значения 'y', а 'cmap' — цветовая схема
scatter = plt.scatter(X2["Humidity"], X2["Wind Speed (km/h)"], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel("Влажность")
plt.ylabel("Скорость ветра (км/ч)")
plt.title("Цветовая карта: Влажность, скорость ветра и ощущаемая температура")
# Добавляем цветовую шкалу с подписью
cbar = plt.colorbar(scatter)
cbar.set_label("Ощущаемая температура (°C)")
plt.show()
'''
[ Выводы ]
Чем выше скорость ветра, тем ниже ощущается температура
Чем выше влажность, тем ниже ощущается температура
'''

# Консольный тестовый стенд
def predict_temperature(humidity: float, wind_speed: float = None) -> float:
    if wind_speed is None:
        # Создаём DataFrame с одной колонкой "Humidity"
        X_input = pd.DataFrame({"Humidity": [humidity]})
        prediction = model.predict(X_input)[0]
    else:
        # Создаём DataFrame с двумя колонками "Humidity" и "Wind Speed (km/h)"
        X_input = pd.DataFrame({
            "Humidity": [humidity],
            "Wind Speed (km/h)": [wind_speed]
        })
        prediction = model2.predict(X_input)[0]
    return prediction

while True:
    user_input = input("Введите влажность (от 0 до 1) или 'q' для выхода: ")    # 0.7
    if user_input.lower() in ["q", "exit"]:
        print("Благодарим за использование наших услуг!")
        break
    try:
        user_humidity = float(user_input)
    except ValueError:
        print("Неверный ввод влажности. Попробуйте ещё раз.")
        continue
    user_wind_speed = input("Введите скорость ветра [или оставьте поле пустым] (км/ч): ")   # 18
    if user_wind_speed.strip() == "":
        temp_pred = predict_temperature(user_humidity)
    else:
        try:
            temp_pred = predict_temperature(user_humidity, float(user_wind_speed))
        except ValueError:
            print("Неверный ввод скорости ветра. Попробуйте ещё раз.")
            continue
    print(f"Предсказанная ощущаемая температура: {temp_pred:.2f}°C")

'''
[ результат ]
Введите влажность (от 0 до 1): 0.7
Введите скорость ветра [или оставьте поле пустым] (км/ч): 18
Предсказанная ощущаемая температура: 9.86°C
'''