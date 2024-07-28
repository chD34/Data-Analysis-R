import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef, \
    balanced_accuracy_score, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# 1.  Відкрити та зчитати наданий файл з даними.
# 2.  Визначити та вивести кількість записів.
# 3.  Видалити атрибут Class.
# 4.  Вивести атрибути, що залишилися.
# 5.  Використовуючи функцію KMeans бібліотеки scikit-learn, виконати
# розбиття набору даних на кластери з випадковою початковою
# ініціалізацією і вивести координати центрів кластерів.
# Оптимальну кількість кластерів визначити на основі початкового
# набору даних трьома різними способами:
#     1) elbow method;
#     2) average silhouette method;
#     3)  prediction  strength  method  (див.  п.  9.2.3  Determining  the  Number  of
# Clusters  книжки  Andriy  Burkov.  The  Hundred-Page  Machine  Learning
# Book).
# Отримані  результати  порівняти  і  пояснити,  який  метод  дав  кращий
# результат і чому так (на Вашу думку).
# 6.  За раніш обраної кількості кластерів багаторазово проведіть
# кластеризацію методом k-середніх, використовуючи для початкової
# ініціалізації метод k-means++.
# Виберіть  найкращий  варіант  кластеризації.  Який  кількісний  критерій
# Ви обрали для відбору найкращої кластеризації?
# 7.  Використовуючи функцію AgglomerativeClustering бібліотеки scikit-
# learn, виконати розбиття набору даних на кластери. Кількість кластерів
# обрати такою ж самою, як і в попередньому методі. Вивести
# координати центрів кластерів.
# 8.  Порівняти результати двох використаних методів кластеризації.

# 1. reading file

csv_file = pandas.read_csv('dataset2_l4.txt')
print('         Task 1 \nRead')
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# 2. count rows
n = len(csv_file.axes[0])
print('\n         Task 2 \nNumber of rows:', n)

# 3-4. deleting Class
csv_file = csv_file.drop(['Class'], axis=1)
print('\n         Task 3-4\n' + str(list(csv_file.columns)))
print('Class deleted!')

# 5. kmeans, elbow and etc.
print('\n         Task 5\n')

# elbow
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    kmeans.fit(csv_file)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

print('Elbow method gives 2 clusters number')

# average silhouette method
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    kmeans.fit(csv_file)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(csv_file, labels))

plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Average Silhouette Method')
plt.show()

print('Average silhouette method also gives 2 clusters number')

# prediction  strength  method


def prediction_strength(data, n_clusters, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=random_state)
    kmeans.fit(data)
    lab = kmeans.labels_
    r2 = r2_score(lab, kmeans.transform(data).min(axis=1))
    predict_strength = 1 - r2
    return predict_strength, kmeans


strengths = []
for i in range(2, 11):
    strength = prediction_strength(csv_file, n_clusters=i)[0]
    strengths.append(strength)

plt.plot(range(2, 11), strengths)
plt.xlabel('Number of clusters')
plt.ylabel('Prediction Strength')
plt.title('Prediction Strength')
plt.show()

print('Prediction  strength  method also gives 2 clusters number\n')
kmeans = KMeans(n_clusters=2, init='random', n_init='auto').fit(csv_file)
csv_file['cluster'] = kmeans.labels_
# print(csv_file.head(10))

centroids = kmeans.cluster_centers_

for i, centroid in enumerate(centroids):
    print(f"Центр кластеру {i+1}: {centroid}")


# 6
print('\n         Task 6\n')


def perform_kmeans(data, num_clusters, num_repeats):
    best_silhouette_score = -1
    best_labels = None

    for _ in range(num_repeats):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto', random_state=1)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_labels = labels

    return best_labels


best_labels = perform_kmeans(csv_file, 2, 20)
silhouette = silhouette_score(csv_file, best_labels)

print("Best Silhouette Score:", silhouette)
print("Best clustering:", best_labels)

# 7
print('\n         Task 7\n')
clustering = AgglomerativeClustering().fit(csv_file)
labels = clustering.fit_predict(csv_file)

silhouette = silhouette_score(csv_file, clustering.labels_)
print("Silhouette Score:", silhouette)

cluster_centers = []
for cluster_label in range(2):
    cluster_points = np.array(csv_file)[labels == cluster_label]
    cluster_center = np.mean(cluster_points, axis=0)
    cluster_centers.append(cluster_center)

print("Cluster Centers:")
for cluster_center in cluster_centers:
    print(cluster_center)

print('\n         Task 8\n')
print('Порівняння 6 і 7 пункту - Методи дали однакові результати')
print('Пункт 6', silhouette_score(csv_file, best_labels))
print('Пункт 7', silhouette)
print('Порівняння 5 і 7 пункту - Методи дали різні результати')
print('Пункт 5', silhouette_score(csv_file, kmeans.labels_))
print('Пункт 7', silhouette)