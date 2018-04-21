# coding: utf-8
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# mnist contains 60,000 training data and 10,000 testing data of hand-writing number


def plot_image(image):
    fig = plt.gcf()  # Initialize
    fig.set_size_inches(10, 10)
    plt.imshow(image, cmap='binary')
    plt.show()


def plot_images_labels_prediction(images, labels, prediction, index, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    num = 25 if num > 25 else num
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[index], cmap='binary')
        title = "label = " + str(images[index])
        title += ", prediction = " + str(prediction[index]) if len(prediction) > 0 else ""
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()


# Return 4 arrays
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 28x28 -> 784x1
X_train_reshaped = X_train.reshape(60000, 784).astype('float32')
X_test_reshaped = X_test.reshape(10000, 784).astype('float32')

# Normalize, convert 0~255 to 0~1
X_train_normalized = X_train_reshaped / 255
X_test_normalized = X_test_reshaped / 255

# Convert labels to one-hot encoding
y_train_one_hot = np_utils.to_categorical(y_train)
y_test_one_hot = np_utils.to_categorical(y_test)

print(len(X_train))
print(len(X_test))
print(len(X_train.shape))  # 60000x28x28
print(len(X_test.shape))  # 10000x28x28

# plot_image(X_train[5])
# print(y_train[5])

# plot_images_labels_prediction(X_train, y_train, [], 0, 10)
# Visualize training process


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


model = Sequential()
# Add layers
model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
print(model.summary())
# Model settings (784 -> 1000 -> 10)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train_normalized,
                          y=y_train_one_hot,
                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)

# Overfitting
show_train_history(train_history, 'acc', 'val_acc')

model_dropout = Sequential()
# Add layers
model_dropout.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model_dropout.summary()
# Model settings (784 -> 1000 -> 10)
model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history_dropout = model_dropout.fit(x=X_train_normalized,
                                          y=y_train_one_hot,
                                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)
model_dropout.summary()
# Model settings (784 -> 1000 -> 10)
model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history_dropout = model_dropout.fit(x=X_train_normalized,
                                          y=y_train_one_hot,
                                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)
show_train_history(train_history_dropout, 'acc', 'val_acc')
