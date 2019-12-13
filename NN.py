import pandas as pd

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# NN models
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=3)

sc = MinMaxScaler()
sc.fit(train)

train = sc.transform(train)
test = sc.transform(test)

hung_test = sc.transform(hung)

def build_dnn(optimizer='adam'):
    dnn = Sequential()
    dnn.add(Dense(units=64, activation='relu', input_shape=(10,)))
    dnn.add(Dense(units=32, activation='relu'))
    dnn.add(Dense(units=16, activation='relu'))
    dnn.add(Dense(units=1, activation='sigmoid'))
    dnn.summary()
    dnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return dnn

def dnn_train(train, target):
    opt = optimizers.Adam(lr=0.001)
    dnn = build_dnn(opt)
    earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath='best.h5',
		                     verbose=1,
		                     save_best_only=True,
		                     monitor='val_loss',
		                     mode='min')
    dnn.fit(train, target,
		epochs = 100,
		batch_size = 64,
		validation_split = 0.15,
		callbacks = [earlystopping,checkpoint])
    return dnn

dnn = dnn_train(train, target)
dnn_prediction_test = dnn.predict(test)
dnn_prediction_test = (dnn_prediction_test > 0.5)*1
acc_test_ann1 = round(metrics.accuracy_score(target_test, dnn_prediction_test) * 100, 2)
print('Test on original dataset (10 features) : ')
print(find_scores(dnn_prediction_test, target_test))

dnn_prediction_test_2 = dnn.predict(hung_test)
dnn_prediction_test_2 = (dnn_prediction_test_2 > 0.5)*1
acc_test_ann2 = round(metrics.accuracy_score(hung_target, dnn_prediction_test_2) * 100, 2)
print('Test on another dataset (10 features) : ')
print(find_scores(dnn_prediction_test_2, hung_target))
