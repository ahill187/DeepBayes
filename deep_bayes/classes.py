class MultiClassifier:
          def __init__(self):
                    self.epochs = 10000
                    self.lr = 0.001
                    self.batch = 1000
                    self.metrics = ['acc', 'mse', 'mae']
                    self.loss = 'categorical_crossentropy'

class Data:
          def __init__(self, length, x, y, y_weights):
                    self.length = length
                    self.y = y
                    self.x = x
                    self.y_weights = y_weights

