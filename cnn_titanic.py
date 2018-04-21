# coding: utf-8
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

filepath = 'data/titanic3.xls'
all_df = pd.read_excel(filepath)
all_df[:10]
all_df.isnull().sum()
cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']
selected_df = all_df[cols]
selected_df.isnull().sum()
selected_df['age'] = selected_df['age'].fillna(selected_df['age'].mean())
selected_df['embarked'] = selected_df['embarked'].fillna('S')
selected_df.isnull().sum()
selected_df['sex'] = selected_df['sex'].map({'female': 0, 'male': 1}).astype(int)
selected_df = pd.get_dummies(data=selected_df, columns=['embarked'])
selected_df[:10]
msk = np.random.rand(len(selected_df)) < 0.8
train_df = selected_df[msk]
test_df = selected_df[~msk]
print('Total: ' + str(len(selected_df)))
print('Train: ' + str(len(train_df)))
print('Test : ' + str(len(test_df)))
train_labels = train_df.values[:, 0]
train_features = train_df.values[:, 1:]
test_labels = test_df.values[:, 0]
test_features = test_df.values[:, 1:]
minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
train_features = minmax_scale.fit_transform(train_features)
test_features = minmax_scale.fit_transform(test_features)
train_features.shape
model = Sequential()
model.add(Dense(units=100, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='relu'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_features, y=train_labels, validation_split=0.15, epochs=100, batch_size=100, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')
scores = model.evaluate(x=test_features, y=test_labels)
scores[1]
