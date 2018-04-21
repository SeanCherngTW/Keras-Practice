# coding: utf-8
import re
import os
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    path = 'data/aclImdb/'
    file_list = []

    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('Read', filetype, 'files', len(file_list))

    # In file_list:
    # Index:     0~12499 = positive
    # Index: 12500~24999 = negative
    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_texts, all_labels


text_train, y_train = read_files('train')
text_test, y_test = read_files('test')
token = Tokenizer(num_words=5000)
token.fit_on_texts(text_train)
X_train_seq = token.texts_to_sequences(text_train)
X_test_seq = token.texts_to_sequences(text_test)
X_train = sequence.pad_sequences(X_train_seq, maxlen=300)
X_test = sequence.pad_sequences(X_test_seq, maxlen=300)
print(len(X_train_seq[104]))
print(len(X_train[104]))
print(len(X_train_seq[6]))
print(len(X_train[1]))
print((X_train[6]))

model = Sequential()
model.add(Embedding(output_dim=32, input_dim=5000, input_length=300))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')
scores = model.evaluate(X_test, y_test, verbose=1)
scores[1]
