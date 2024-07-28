import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef, \
    balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

'''
1. Відкрити та зчитати наданий файл з даними.
2. Визначити та вивести кількість записів.
3. Вивести атрибути набору даних.
4. З’ясувати збалансованість набору даних.
5. Отримати двадцять варіантів перемішування набору даних та 
розділення його на навчальну ( тренувальну) та тестову вибірки, 
використовуючи функцію ShuffleSplit. Сформувати начальну та тестові 
вибірки на основі обраного користувачем варіанту .
6. Використовуючи  функцію  KNeighborsClassifier  бібліотеки  scikit-learn, 
збудувати класифікаційну модель на основі методу k найближчих 
сусідів  (кількість сусідів обрати самостійно, вибір аргументувати)  та 
навчити її на тренувальній вибірці, вважаючи, що цільова 
характеристика визначається стовпчиком Class, а всі інші виступають в 
ролі вихідних аргументів.
7. Обчислити класифікаційні метрики збудованої моделі для тренувальної
та тестової вибірки. Представити результати роботи моделі на тестовій
вибірці графічно.
8. Обрати  алгоритм  KDTree  та  з’ясувати  вплив  розміру  листа  (від  20  до 
200  з  кроком 5) на  результати  класифікації.  Результати  представити 
графічно.

'''

# 1. reading file

csv_file = pandas.read_csv('dataset2_l4.txt')
print('         Task 1 \nRead')
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# 2-3. count rows and columns
n, m = len(csv_file.axes[0]), len(csv_file.axes[1])
print('\n         Task 2-3 \nNumber of rows:', n)
print('Number of columns', m)

# 4. balancing
print('\n         Task 4')
print('Number of objects each class:\n', csv_file.groupby(['Class'])['Class'].count())
print('\nSo, dataset is not balanced')

# 5. dividing into train and test samples
elements = csv_file
variants = ShuffleSplit(n_splits=20, train_size=0.7)


print('\n         Task 5')


print('\nChoose one of the random variants. Print num from 1 to 20: ')
while True:
    try:
        number = input()
        if (number.isdigit()) and (int(number) <= 20) and (int(number) >= 1):
            break
        else:
            raise ValueError
    except ValueError:
        print('Print num from 1 to 20!')
        continue

counter = 1

for train_data, test_data in variants.split(elements):
    counter += 1
    if counter == int(number)+1:
        train = list(train_data)
        test = list(test_data)

train_df = csv_file.loc[train]
test_df = csv_file.loc[test]

print('\n         Task 6-7')
# 6-7
x_test = test_df[["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]]
x_train = train_df[["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]]
y_test = test_df["Class"]
y_train = train_df["Class"]
neigh = KNeighborsClassifier(n_neighbors=4)
# (2-4)

neigh.fit(x_train, y_train)

test_predict = neigh.predict(x_test)
train_predict = neigh.predict(x_train)

y_train = list(y_train)
y_test = list(y_test)
metrics_train = [accuracy_score(y_train, train_predict), precision_score(y_train, train_predict, average="macro"),
                recall_score(y_train, train_predict, average="macro"), fbeta_score(y_train, train_predict, beta=0.5
                                                                                   , average="macro"),
                fbeta_score(y_train, train_predict, beta=1, average="macro"), fbeta_score(y_train, train_predict,
                                                                                          beta=2, average="macro"),
                matthews_corrcoef(y_train, train_predict), balanced_accuracy_score(y_train, train_predict),
                balanced_accuracy_score(y_train, train_predict, adjusted=True)]

metrics_test = [accuracy_score(y_test, test_predict), precision_score(y_test, test_predict, average="macro"),
                recall_score(y_test, test_predict, average="macro"), fbeta_score(y_test, test_predict, beta=0.5,
                                                                                 average="macro"),
                fbeta_score(y_test, test_predict, beta=1, average="macro"), fbeta_score(y_test, test_predict, beta=2,
                                                                                        average="macro"),
                matthews_corrcoef(y_test, test_predict), balanced_accuracy_score(y_test, test_predict),
                balanced_accuracy_score(y_test, test_predict, adjusted=True)]

print('\t\t\t\t\t\ttrain\t\t\t\ttest')
print('Accuracy: \t\t\t', metrics_train[0], '\t', metrics_test[0],
      '\nPrecision: \t\t\t', metrics_train[1], '\t', metrics_test[1],
      '\nRecall: \t\t\t', metrics_train[2], '\t', metrics_test[2],
      '\nF-Beta score(0.5): \t', metrics_train[3], '\t', metrics_test[3],
      '\nF-Beta score(1): \t', metrics_train[4], '\t', metrics_test[4],
      '\nF-Beta score(2): \t', metrics_train[5], '\t', metrics_test[5],
      '\nMatthews coef.: \t', metrics_train[6], '\t', metrics_test[6],
      '\nBalanced Accuracy: \t', metrics_train[7], '\t', metrics_test[7],
      '\nYoudenJStatistic: \t', metrics_train[8], '\t', metrics_test[8])

metrics_name = ['Accuracy', 'Precision', 'Recall', 'F-Beta (0.5)', 'F-Beta (1)', 'F-Beta (2)',
                'Matthews', 'Balanced Acc', 'YoudenJStat']

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter([i for i in metrics_name], metrics_test, label='Test', color="violet")
ax.legend()
plt.title('Metrics of test sample')
plt.show()
print('\nGraphics built!')

# task 8

print('\n         Task 8')
leaf_size = [i for i in range(20, 205, 5)]
plt.figure(figsize=(10, 6))
for el in leaf_size:
    neigh = KNeighborsClassifier(n_neighbors=4, leaf_size=el, algorithm="kd_tree")
    neigh.fit(x_train, y_train)

    test_predict = neigh.predict(x_test)
    train_predict = neigh.predict(x_train)

    metrics_train = [accuracy_score(y_train, train_predict), precision_score(y_train, train_predict, average="macro"),
                     recall_score(y_train, train_predict, average="macro"), fbeta_score(y_train, train_predict, beta=0.5
                                                                                        , average="macro"),
                     fbeta_score(y_train, train_predict, beta=1, average="macro"), fbeta_score(y_train, train_predict,
                                                                                               beta=2, average="macro"),
                     matthews_corrcoef(y_train, train_predict), balanced_accuracy_score(y_train, train_predict),
                     balanced_accuracy_score(y_train, train_predict, adjusted=True)]

    metrics_test = [accuracy_score(y_test, test_predict), precision_score(y_test, test_predict, average="macro"),
                    recall_score(y_test, test_predict, average="macro"), fbeta_score(y_test, test_predict, beta=0.5,
                                                                                     average="macro"),
                    fbeta_score(y_test, test_predict, beta=1, average="macro"),
                    fbeta_score(y_test, test_predict, beta=2,
                                average="macro"),
                    matthews_corrcoef(y_test, test_predict), balanced_accuracy_score(y_test, test_predict),
                    balanced_accuracy_score(y_test, test_predict, adjusted=True)]

    plt.scatter([i for i in metrics_name], metrics_test, label='Test')
    plt.title('Metrics of test sample')

plt.show()
print("Розмір листя не має значення - всі обчислені метрики однакові")

