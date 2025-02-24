
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------- Здоровье и медицина --------

# 10. Кластеризация на наборе данных о здоровье и медицине

'''
Данные имеют вид:
576161553759760384|Thu Mar 12 23:23:13 +0000 2015|An ultrasound for the brain restores memory in Alzheimer's w/o so much as a needle. But only--so far--in mice.
Уникальный идентификатор (например, твит ID)
Дата и время публикации
Текст публикации (с URL и другой информацией)
* через разделитель | 
'''
optimal_k = 5   # оптимальное число кластеров
news = 5        # кол-во новостей для каждого кластера (для вывода примера)

# Определяем словарь с именами файлов и их кодировками
files = {
    'bbchealth.txt': 'utf-8',
    'cbchealth.txt': 'utf-8',
    'cnnhealth.txt': 'mac_roman',         # для MacRoman используем 'mac_roman'
    'everydayhealth.txt': 'utf-8',
    'foxnewshealth.txt': 'windows-1252',
    'gdnhealthcare.txt': 'mac_roman',
    'goodhealth.txt': 'mac_roman',
    'KaiserHealthNews.txt': 'windows-1252',
    'latimeshealth.txt': 'utf-8',
    'msnhealthnews.txt': 'windows-1252',
    'NBChealth.txt': 'windows-1252',
    'nprhealth.txt': 'utf-8',
    'nytimeshealth.txt': 'utf-8',
    'reuters_health.txt': 'utf-8',
    'usnewshealth.txt': 'utf-8',
    'wsjhealth.txt': 'windows-1252'
}

# все файлы лежат в папке "Health-Tweets"!
data_dir = 'Health-Tweets'

# Чтение файлов и объединение строк в единый DataFrame
rows = []
for filename, encoding in files.items():
    file_path = os.path.join(data_dir, filename)
    try:
        with open(file_path, encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Разбиваем строку по разделителю "|", maxsplit=2 позволяет сохранить текст как единое поле
                parts = line.split('|', maxsplit=2)
                if len(parts) != 3:
                    continue
                tweet_id, dt_str, text = parts
                rows.append({'id': tweet_id, 'datetime': dt_str, 'text': text})
    except Exception as e:
        print(f'Ошибка при чтении файла {filename}: {e}')

df = pd.DataFrame(rows)
#print("Размер объединённого DataFrame:", df.shape)
# Размер объединённого DataFrame: (63326, 3)

# Преобразуем поле даты в формат datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
#print("Преобразование даты завершено.")

# Очистка текстовых данных: удаляем URL-адреса
def remove_urls(text):
    return re.sub(r"http\S+", '', text)  # Регулярное выражение для поиска URL
df['clean_text'] = df['text'].apply(remove_urls)
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
#print("Очистка текстовых данных завершена.")

# Векторизация текстов с помощью TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
#print("TF-IDF векторизация завершена. Размер матрицы признаков:", X.shape)
# TF-IDF векторизация завершена. Размер матрицы признаков: (63326, 30700)

# Определение оптимального числа кластеров методом локтя с помощью KMeans
sss = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    sss.append(km.inertia_)
'''
plt.figure(figsize=(8, 5))
plt.plot(K, sss, 'bx-')
plt.xlabel('Количество кластеров (k)')
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.show()
'''

# Алгоритм k-средних 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters
print("Кластеризация завершена. Распределение по кластерам:")
print(df['cluster'].value_counts())
'''
Кластеризация завершена. Распределение по кластерам:
cluster
0    59848
1     2310
2     1168

или 

cluster
3    53843
4     4380
1     2292
0     1648
2     1163
'''
# Визуализация результатов с помощью PCA - Метода главных компонент
pca = PCA(n_components=2, random_state=42)
X_dense = X.toarray()
X_reduced = pca.fit_transform(X_dense)

#'''
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=10)
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.title('Визуализация кластеров (PCA)')
plt.colorbar(label='Кластер')
plt.show()
#'''

# Анализ кластеров: выводим несколько примеров публикаций для каждого кластера
for i in range(optimal_k):
    print(f"\nПримеры публикаций для кластера {i}:")
    sample_texts = df[df['cluster'] == i]['clean_text'].head(news).tolist()
    for text in sample_texts:
        print(" -", text)
'''
[ Вывод ]

Примеры публикаций для кластера 0:  # всё остальное
 - GP workload harming care - BMA poll
 - Short people's 'heart risk greater'
 - New approach against HIV 'promising'        
 - Coalition 'undermined NHS' - doctors        
 - Review of case against NHS manager

Примеры публикаций для кластера 1:  # про рак
 - Breast cancer risk test devised
 - VIDEO: Skin cancer spike 'from 60s holidays'
 - Skin cancer 'linked to holiday boom'        
 - Personal cancer vaccines 'exciting'
 - Fitness linked to lower cancer risk

Примеры публикаций для кластера 2:  # про детей
 - Call to ban energy drinks for kids
 - Later sunsets 'make kids more active'
 - Schools 'should check kids' teeth'
 - Treats in moderation make kids happy
 - Mom of 7 'spooked' by vaccinations reverses stand — but then kids get sick


 или

 Примеры публикаций для кластера 0:
 - Campaigners make Men B vaccine plea
 - VIDEO: What can make you happy?
 - Law to make FGM reporting mandatory
 - Google introduces illness tips
 - Partners can 'make pain worse'
 - Scientists make 'feel full' chemical
 - Scientists make enzymes from scratch

Примеры публикаций для кластера 1:
 - Breast cancer risk test devised
 - VIDEO: Skin cancer spike 'from 60s holidays'
 - Skin cancer 'linked to holiday boom'
 - Personal cancer vaccines 'exciting'
 - Fitness linked to lower cancer risk
 - Preventive surgery for cancer genes
 - VIDEO: Could cannabis oil cure cancer?

Примеры публикаций для кластера 2:
 - Call to ban energy drinks for kids
 - Later sunsets 'make kids more active'
 - Schools 'should check kids' teeth'
 - Treats in moderation make kids happy
 - Mom of 7 'spooked' by vaccinations reverses stand — but then kids get sick
 - Washing dishes by hand linked to fewer allergies in kids
 - RT @kimbrunhuber: Hear from mom of immunocompromised infant who can't get the measles shot why herd immunity is so important for *all* kids…

Примеры публикаций для кластера 3:
 - GP workload harming care - BMA poll 
 - Short people's 'heart risk greater' 
 - New approach against HIV 'promising'
 - Coalition 'undermined NHS' - doctors
 - Review of case against NHS manager  
 - VIDEO: 'All day is empty, what am I going to do?'
 - VIDEO: 'Overhaul needed' for end-of-life care

Примеры публикаций для кластера 4:
 - Guinea declares Ebola 'emergency'
 - VIDEO: Military healthcare worker free of Ebola
 - British medic declared free of Ebola
 - Sierra Leone in Ebola lockdown
 - Ebola 'more deadly' in young children
 - Has Ebola focus led to other killer diseases being ignored?
 - VIDEO: Charity slams global Ebola response
PS X:\Desktop\XPythonProject\ML> 
'''

