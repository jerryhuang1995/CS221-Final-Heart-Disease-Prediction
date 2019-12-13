import pandas as pd

# preprocessing
from sklearn.model_selection import train_test_split

# models
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

def find_scores(predict, target, NN=True, test=None):
    score_dict = {}
    TP, FP, P = 0, 0, 0
    
    if NN:
        score_dict['accuracy'] = round(metrics.accuracy_score(target, predict) * 100, 2)
    else:
        score_dict['accuracy'] = round(logreg.score(test, target) * 100, 2) 
        predict = [[pred] for pred in predict]

    tar = []
    for tg in target:
        tar.append(tg)

    for i in range(len(predict)):
        if tar[i] == 1:
            P += 1
        if predict[i][0] == 1:
            if tar[i] == 1:
                TP += 1
            else:
                FP += 1

    score_dict['precision'] =  TP / (TP + FP) 
    score_dict['recall'] = TP / P
    score_dict['f1_score'] = 2 * (score_dict['recall'] * score_dict['precision']) / (score_dict['recall'] + score_dict['precision'])

    return score_dict

data = pd.read_csv('combine.csv')
hung = pd.read_csv('new.csv')

target_name = 'target'

data_target = data[target_name]
data = data.drop([target_name, 'slope', 'ca', 'thal'], axis=1)
hung_target = hung[target_name]
hung = hung.drop([target_name, 'slope', 'ca', 'thal'], axis=1)

for i in range(len(data_target)):
    if data_target[i] > 1:
        data_target[i] = 1

for i in range(len(hung_target)):
    if hung_target[i] > 1:
        hung_target[i] = 1

train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=2)

sc = MinMaxScaler()
sc.fit(train)

train = sc.transform(train)
test = sc.transform(test)

hung_test = sc.transform(hung)

logreg = LogisticRegression()
logreg.fit(train, target)
acc_log = round(logreg.score(train, target) * 100, 2)
acc_test_log = round(logreg.score(test, target_test) * 100, 2)
print('coefficients :')
print(logreg.coef_)

print('test on original set (10 features) : ')
print(find_scores(logreg.predict(test), target_test, False, test))

acc_log = round(logreg.score(train, target) * 100, 2)
acc_test_log = round(logreg.score(hung_test, hung_target) * 100, 2)

print('test on new set (10 features) : ')
print(find_scores(logreg.predict(hung_test), hung_target, False, hung_test))
