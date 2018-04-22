# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0
y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)
print(X_train_normalized.shape)
print(X_test_normalized.shape)
print(y_train_one_hot.shape)
print(y_test_one_hot.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train_normalized, y=y_train_one_hot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'acc', 'val_acc')
model.save_weights("cifarCnnModel.h5")
predicted_probability = model.predict(X_test_normalized)
for i in range(10):
    print(predicted_probability[0][i])
label_dict = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: "Cat", 4: "Deer",
              5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'truck'}

# y_head[i] == The label of testing data in index i, 10 dimensions, one-hot encoding
# y_head[i][j] == 0 or 1


def show_predicted_probability(y_head, y, x, predicted_prob, i):
    print('label   : ', label_dict[y_head[i][0]])
    print('predict : ', label_dict[y[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print("{0:<10} Probability: {1:.3f}".format(label_dict[j], predicted_prob[i][j]))


prediction = model.predict_classes(X_test_normalized)
show_predicted_probability(y_test, prediction, X_test_normalized, predicted_probability, 210)
scores = model.evaluate(X_test_normalized, y_test_one_hot, verbose=2)
print(scores[1])
