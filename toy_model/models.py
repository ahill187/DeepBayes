from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Nadam
from keras.callbacks import EarlyStopping
import keras.regularizers as regularizers

class MultiClassifier:
          def __init__(self):
                    self.epochs = 10000
                    self.lr = 0.001
                    self.batch = 1000
                    self.metrics = ['acc', 'mse', 'mae']
                    self.loss = 'categorical_crossentropy'


def Model(multi, x_train, y_train, bins, reg):
    """Keras neural network.

    Args:
        multi: TODO
        x_train (np.array): An n x 1 array of real numbers.
        y_train (np.array): An n x bins array of real numbers, one-hot encoded.
        bins (int): number of output bins.
        reg: TODO
    """
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=(1,)))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(bins, activation='softmax', kernel_regularizer=regularizers.l2(reg)))
    nadam = Nadam(lr = multi.lr)
    stop = EarlyStopping(patience=5)
    model.compile(loss=multi.loss, optimizer=nadam, metrics=multi.metrics)
    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch, callbacks=[stop])
    return model
