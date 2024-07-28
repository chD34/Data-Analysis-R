import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef, \
    balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import matplotlib.pyplot as plt

# 1. reading file

csv_file = pandas.read_csv('dataset_2.txt', header=None)
print('         Task 1 \nRead')

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# 2. count rows and columns
n, m = len(csv_file.axes[0]), len(csv_file.axes[1])
print('\n         Task 2 \nNumber of rows:', n)
print('Number of columns', m)

# 3. printing first 10 rows
print('\n         Task 3 \nFirst 10 rows:\n', csv_file.head(10))

# 4. split into educational (training) and test samples
print('\n         Task 4\nSplit into train and test samples(70/30).\n')
features = list(csv_file.columns[2:7])

y = csv_file[7]
X = csv_file[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
train, test = train_test_split(csv_file, test_size=0.3)

n = len(X_test.axes[0])
print('Number of rows(test):', n)
n = len(X_train.axes[0])
print('Number of rows(train):', n)

# 5-6. Decision tree

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True,
                           feature_names=X.columns, class_names=['0', "1"])

graph = graphviz.Source(dot_data, format='png')
graph.render("decision_tree_graphivz")
print('\n         Task 5-6 \nDecision tree builded and file with decision tree created!')

# 7 Metrics
print('\n         Task 7 ')

test_predict = clf.predict(X_test)
train_predict = clf.predict(X_train)

metrics_train = [accuracy_score(y_train, train_predict), precision_score(y_train, train_predict),
                 recall_score(y_train, train_predict), fbeta_score(y_train, train_predict, beta=0.5),
                 fbeta_score(y_train, train_predict, beta=1), fbeta_score(y_train, train_predict, beta=2),
                 matthews_corrcoef(y_train, train_predict), balanced_accuracy_score(y_train, train_predict),
                 balanced_accuracy_score(y_train, train_predict, adjusted=True), roc_auc_score(y_train, train_predict),
                 average_precision_score(y_train, train_predict)]

metrics_test = [accuracy_score(y_test, test_predict), precision_score(y_test, test_predict),
                recall_score(y_test, test_predict), fbeta_score(y_test, test_predict, beta=0.5),
                fbeta_score(y_test, test_predict, beta=1), fbeta_score(y_test, test_predict, beta=2),
                matthews_corrcoef(y_test, test_predict), balanced_accuracy_score(y_test, test_predict),
                balanced_accuracy_score(y_test, test_predict, adjusted=True), roc_auc_score(y_test, test_predict),
                average_precision_score(y_test, test_predict)]
metrics_name = ['Accuracy', 'Precision', 'Recall', 'F-Beta (0.5)', 'F-Beta (1)', 'F-Beta (2)',
                'Matthews coef', 'Balanced Acc', 'YoudenJStat', 'ROC AUC', 'PR AUC']

print('    Metrics for train/test sample(gini)')
print('\t\t\t\t\t\ttrain\t\t\t\ttest')
print('Accuracy: \t\t\t', metrics_train[0], '\t', metrics_test[0],
      '\nPrecision: \t\t\t', metrics_train[1], '\t', metrics_test[1],
      '\nRecall: \t\t\t', metrics_train[2], '\t', metrics_test[2],
      '\nF-Beta score(0.5): \t', metrics_train[3], '\t', metrics_test[3],
      '\nF-Beta score(1): \t', metrics_train[4], '\t\t', metrics_test[4],
      '\nF-Beta score(2): \t', metrics_train[5], '\t', metrics_test[5],
      '\nMatthews coef.: \t', metrics_train[6], '\t', metrics_test[6],
      '\nBalanced Accuracy: \t', metrics_train[7], '\t\t', metrics_test[7],
      '\nYoudenJStatistic: \t', metrics_train[8], '\t', metrics_test[8],
      '\nROC AUC: \t\t\t', metrics_train[9], '\t', metrics_test[9],
      '\nPR AUC: \t\t\t', metrics_train[10], '\t', metrics_test[10])

# GRAPHIC true/false answers
tp, tn, fp, fn = 0, 0, 0, 0
true_answers, false_answers = 0, 0
for i in range(len(y_test)):
    if list(y_test)[i] == test_predict[i]:
        if list(y_test)[i] == 0:
            tn += 1
        else:
            tp += 1
        true_answers += 1
    else:
        if test_predict[i] == 0:
            fn += 1
        else:
            fp += 1
        false_answers += 1

plt.bar(['True answers', 'False answers'], [true_answers, false_answers], width=0.5, color='violet')
plt.ylabel("Number")
plt.text('True answers', true_answers, f'{true_answers} (tn: {tn}, tp: {tp})', horizontalalignment='center',
         verticalalignment='bottom')
plt.text('False answers', false_answers, f'{false_answers} (fn: {fn}, fp: {fp})', horizontalalignment='center',
         verticalalignment='bottom')
plt.title('Result of work on test sample(gini)')
plt.show()

# GRAPHIC metrics(gini)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(metrics_test, [i for i in metrics_name], label='Test')
ax.scatter(metrics_train, [el for el in metrics_name], label='Train')
ax.legend()
plt.title('Comparison metrics of test and train samples')
plt.show()
print('Graphics built!')

# entropy criterion
print('\nTry to use entropy criterion to build tree')
clf2 = DecisionTreeClassifier(max_depth=5, criterion='entropy')
clf2.fit(X_train, y_train)
test_predict = clf2.predict(X_test)
train_predict = clf2.predict(X_train)
dot_data2 = export_graphviz(clf2, out_file=None, filled=True, rounded=True,
                            feature_names=X.columns, class_names=['0', "1"])
graph2 = graphviz.Source(dot_data2, format='png')
graph2.render("decision_tree_graphivz2")


metrics_train_2 = [accuracy_score(y_train, train_predict), precision_score(y_train, train_predict),
                 recall_score(y_train, train_predict), fbeta_score(y_train, train_predict, beta=0.5),
                 fbeta_score(y_train, train_predict, beta=1), fbeta_score(y_train, train_predict, beta=2),
                 matthews_corrcoef(y_train, train_predict), balanced_accuracy_score(y_train, train_predict),
                 balanced_accuracy_score(y_train, train_predict, adjusted=True), roc_auc_score(y_train, train_predict),
                 average_precision_score(y_train, train_predict)]

metrics_test_2 = [accuracy_score(y_test, test_predict), precision_score(y_test, test_predict),
                recall_score(y_test, test_predict), fbeta_score(y_test, test_predict, beta=0.5),
                fbeta_score(y_test, test_predict, beta=1), fbeta_score(y_test, test_predict, beta=2),
                matthews_corrcoef(y_test, test_predict), balanced_accuracy_score(y_test, test_predict),
                balanced_accuracy_score(y_test, test_predict, adjusted=True), roc_auc_score(y_test, test_predict),
                average_precision_score(y_test, test_predict)]

# GRAPHIC entropy
tp, tn, fp, fn = 0, 0, 0, 0
true_answers, false_answers = 0, 0
for i in range(len(y_test)):
    if list(y_test)[i] == test_predict[i]:
        if list(y_test)[i] == 0:
            tn += 1
        else:
            tp += 1
        true_answers += 1
    else:
        if test_predict[i] == 0:
            fn += 1
        else:
            fp += 1
        false_answers += 1

plt.bar(['True answers', 'False answers'], [true_answers, false_answers], width=0.5, color='green')
plt.ylabel("Number")
plt.text('True answers', true_answers, f'{true_answers} (tn: {tn}, tp: {tp})', horizontalalignment='center',
         verticalalignment='bottom')
plt.text('False answers', false_answers, f'{false_answers} (fn: {fn}, fp: {fp})', horizontalalignment='center',
         verticalalignment='bottom')
plt.title('Result of work on test sample(entropy)')
plt.show()

# METRICS entropy
print('    Metrics for train/test sample(entropy)')
print('\t\t\t\t\t\ttrain\t\t\t\ttest')
print('Accuracy: \t\t\t', metrics_train_2[0], '\t\t', metrics_test_2[0],
      '\nPrecision: \t\t\t', metrics_train_2[1], '\t', metrics_test_2[1],
      '\nRecall: \t\t\t', metrics_train_2[2], '\t', metrics_test_2[2],
      '\nF-Beta score(0.5): \t', metrics_train_2[3], '\t', metrics_test_2[3],
      '\nF-Beta score(1): \t', metrics_train_2[4], '\t', metrics_test_2[4],
      '\nF-Beta score(2): \t', metrics_train_2[5], '\t', metrics_test_2[5],
      '\nMatthews coef.: \t', metrics_train_2[6], '\t', metrics_test_2[6],
      '\nBalanced Accuracy: \t', metrics_train_2[7], '\t', metrics_test_2[7],
      '\nYoudenJStatistic: \t', metrics_train_2[8], '\t', metrics_test_2[8],
      '\nROC AUC: \t\t\t', metrics_train_2[9], '\t', metrics_test_2[9],
      '\nPR AUC: \t\t\t', metrics_train_2[10], '\t', metrics_test_2[10])

# GRAPHIC - Comparison metrics of test and train samples(GINI VS ENTROPY)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(metrics_test, [i for i in metrics_name], label='Test (gini)', color='royalblue', marker='>')
ax.scatter(metrics_train, [el for el in metrics_name], label='Train (gini)', color='royalblue')
ax.scatter(metrics_test_2, [i for i in metrics_name], label='Test (entropy)', color='orange', marker='>')
ax.scatter(metrics_train_2, [el for el in metrics_name], label='Train (entropy)', color='orange')
ax.legend()
plt.title('Comparison metrics of test and train samples(GINI VS ENTROPY)')
plt.show()
print('\nДивлячись на графік порівняння метрик для дерев з застосуванням\nкритеріїв розщеплення - ентропію та Джині, '
      'я побачила, \nщо на тренувальній вибірці ентропія дала кращий результат \nпо багатьом метрикам, '
      'порівнюючи з Джині, але \nна тесті переважно гірший, або кращий, але з дуже малим відхиленням')

# Task 8
print('\n         Task 8 ')
clf3 = DecisionTreeClassifier()
clf3.fit(X_train, y_train)
test_predict = clf3.predict(X_test)
train_predict = clf3.predict(X_train)
dot_data3 = export_graphviz(clf3, out_file=None, filled=True, rounded=True,
                feature_names=X.columns, class_names=['0', "1"])
graph3 = graphviz.Source(dot_data3, format='png')
graph3.render("decision_tree_graphivz3")

metrics_train_3 = [accuracy_score(y_train, train_predict), precision_score(y_train, train_predict),
                 recall_score(y_train, train_predict), fbeta_score(y_train, train_predict, beta=0.5),
                 fbeta_score(y_train, train_predict, beta=1), fbeta_score(y_train, train_predict, beta=2),
                 matthews_corrcoef(y_train, train_predict), balanced_accuracy_score(y_train, train_predict),
                 balanced_accuracy_score(y_train, train_predict, adjusted=True), roc_auc_score(y_train, train_predict),
                 average_precision_score(y_train, train_predict)]

metrics_test_3 = [accuracy_score(y_test, test_predict), precision_score(y_test, test_predict),
                recall_score(y_test, test_predict), fbeta_score(y_test, test_predict, beta=0.5),
                fbeta_score(y_test, test_predict, beta=1), fbeta_score(y_test, test_predict, beta=2),
                matthews_corrcoef(y_test, test_predict), balanced_accuracy_score(y_test, test_predict),
                balanced_accuracy_score(y_test, test_predict, adjusted=True), roc_auc_score(y_test, test_predict),
                average_precision_score(y_test, test_predict)]

print('    Metrics for train/test sample(maximum leaves)')
print('\t\t\t\t\t\ttrain\t\t\t\ttest')
print('Accuracy: \t\t\t', metrics_train_3[0], '\t', metrics_test_3[0],
      '\nPrecision: \t\t\t', metrics_train_3[1], '\t', metrics_test_3[1],
      '\nRecall: \t\t\t', metrics_train_3[2], '\t', metrics_test_3[2],
      '\nF-Beta score(0.5): \t', metrics_train_3[3], '\t', metrics_test_3[3],
      '\nF-Beta score(1): \t', metrics_train_3[4], '\t', metrics_test_3[4],
      '\nF-Beta score(2): \t', metrics_train_3[5], '\t', metrics_test_3[5],
      '\nMatthews coef.: \t', metrics_train_3[6], '\t', metrics_test_3[6],
      '\nBalanced Accuracy: \t', metrics_train_3[7], '\t', metrics_test_3[7],
      '\nYoudenJStatistic: \t', metrics_train_3[8], '\t', metrics_test_3[8],
      '\nROC AUC: \t\t\t', metrics_train_3[9], '\t', metrics_test_3[9],
      '\nPR AUC: \t\t\t', metrics_train_3[10], '\t', metrics_test_3[10])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(metrics_test, [i for i in metrics_name], label='Test (depth 5)', color='royalblue', marker='>')
ax.scatter(metrics_train, [el for el in metrics_name], label='Train (depth 5)', color='royalblue')
ax.scatter(metrics_test_3, [i for i in metrics_name], label='Test (max leaves)', color='orange', marker='>')
ax.scatter(metrics_train_3, [el for el in metrics_name], label='Train (max leaves)', color='orange')
ax.legend()
plt.title('Comparison metrics of test and train samples(maximum leaves and up to depth 5)')
plt.show()
print('Порівнюючи метрики, бачимо, що на тренувальній вибірці модель \n'
      'з максимальною кількістю листя і мінімальною кількістю елементів \nв кожному працює ідеально, '
      'всі метрики одинички, \nале що робиться на тестовій вибірці - результати гірші чим, '
      '\nякби ми брали глибину 5(див. графік), тому, можна сказати відбувається перенавчання')

# 9 Feature importances
print('\n         Task 9 ')
plt.bar(features, clf.feature_importances_, width=0.5, color='gold')
plt.title('Feature importances')
plt.show()
print('Метод feature_impotances_ вимірює важливість атрибутів від 0 до 1,\n'
      'припускаю що 0 це те що ознака не використовується в моделі, '
      'а 1 означає, \nщо ознака є дуже важливою, тобто дуже часто використовується.\n'
      'Я думаю, що тут розглядається те, що чим частіше дерево рішень використовує \n'
      'атрибут, тим більша буде його важливість, і відповідно не дуже \nзначимими будуть ті атрибути, '
      'які не досить часто зустрічаються в \nдереві і їхня роль в класифікації не дуже значима')