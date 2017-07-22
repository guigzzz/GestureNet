from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import sys

class GestureNetwork():
    def __init__(self):
        pass

    def build(self, num_classes, input_res, activation = 'relu'):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(input_res, input_res, 3), activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation=activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))
        self.model.add(Dense(128, activation=activation))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x, y, val_split = 0.2, epochs = 50, batch = 100):
        if not self.one_hot(y):
            y = to_categorical(y)
        self.fitted = True
        return self.model.fit(np.asarray(x), np.asarray(y), validation_split = val_split, epochs = epochs, batch_size = batch)

    def save(self, path, class_labels = None):
        self.model.save(path)
        if class_labels:
            from h5py import File, special_dtype
            with File(path, 'r+') as f:
                class_labels = np.array(list(class_labels), dtype = object)
                dt = special_dtype(vlen = str)
                dset = f.create_dataset('labels', data = class_labels, dtype = dt)
                print(dset)

    def load(self, path, load_labels = False):
        self.model = load_model(path)
        self.loaded = True

        if load_labels:
            self.ret_labels = True
            from h5py import File
            with File(path, 'r') as f:
                if 'labels' in f.keys():
                    self.labels = list(f['labels'].value)

    def one_hot(self, vec):
        if type(vec[0]) is list:
            return True
        return False

    def evaluate(self, x, y):
        if not self.one_hot(y):
            y = to_categorical(y)
        return self.model.evaluate(np.asarray(x), y)

    def predict(self, x):
        if self.loaded or self.fitted:
            if len(x.shape) < 4:
                x = [x]
            if self.ret_labels:
                return self.labels[np.argmax(self.model.predict_on_batch(np.asarray(x)))]
            else:
                return self.model.predict_on_batch(np.asarray(x))
        else:
            print('model isnt fitted or a pre-trained model hasnt been loaded')
            sys.exit(1)