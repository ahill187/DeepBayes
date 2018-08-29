from training_base_AH import training_base
from data_helper_functions import *
from classes import Data

#if running locally for >24h you may want to set renewtokens=True

def getTrees(bins, args):
          train = training_base(args=args, testrun=False,renewtokens=False)
          train.train_data.setBins(bins)
          return train

def getModel(train, MODELLIST, custom=False, bin_weights=[]):
          model = int(train.keras_model_method)
          if model not in MODELLIST:
                    raise ValueError('Unknown model %d' % model)
          print('Setting model', model)
          print(MODELLIST[model])
          train.setModel(model=MODELLIST[model]['method'],
                         arch=MODELLIST[model]['arch'],
                         dropoutRate=MODELLIST[model]['dropout'],
                         batchNorm=MODELLIST[model]['batchNorm'],
                         splitInputs=MODELLIST[model]['splitInputs'],
                         pfix=MODELLIST[model]['pfix'],
                         reg=MODELLIST[model]['reg'],
                         bins=MODELLIST[model]['bins'],
                         custom=custom, bin_weights=bin_weights)
          return train

def compileModel(train, multi, MODELLIST):
          metricsList=multi.metrics
          predictionLabels = np.arange(10)
          predictionLabels = [str(x) for x in predictionLabels]
          train.defineCustomPredictionLabels(predictionLabels)
          train.defineCustomPredictionLabels(predictionLabels)
          model = int(train.keras_model_method)
          train.compileModel(learningrate=MODELLIST[model]['learningrate'],
                             loss=MODELLIST[model]['loss'],
                             metrics=metricsList)
          print(train.keras_model.summary())
          return train

def fitModel(train, MODELLIST, custom=False, bin_weights=[], ndata=1000, **fitargs):
          model = int(train.keras_model_method)
          nepochs=MODELLIST[model]['nepochs']
          batchsize=MODELLIST[model]['batchsize']
          nbatches = int(round(ndata / batchsize))
          if nbatches == 0:
              nbatches = 1
          print("Number of Data points = %d" % nbatches*batchsize)
          # model,history = train.trainModel(nepochs=nepochs,
          #                               batchsize=batchsize,
          #                               stop_patience=300, #stop after N-epochs if validation loss increases
          #                               lr_factor=0.5,     #adapt learning rate if validation loss flattens
          #                               lr_patience=15,
          #                               lr_epsilon=0.0001,
          #                               lr_cooldown=2,
          #                               lr_minimum=0.0001,
          #                               maxqsize=100,       #play if file system is unstable
          #                               nbatches=nbatches,
          #                               **fitargs)
          model = train.trainModel(nepochs=nepochs,
                                        batchsize=batchsize,
                                        stop_patience=300, #stop after N-epochs if validation loss increases
                                        lr_factor=0.5,     #adapt learning rate if validation loss flattens
                                        lr_patience=15,
                                        lr_epsilon=0.0001,
                                        lr_cooldown=2,
                                        lr_minimum=0.0001,
                                        maxqsize=5,       #play if file system is unstable
                                        nbatches=nbatches,
                                        custom=custom, bin_weights=bin_weights,
                                        **fitargs)
          return model

def fitModelFast(train, data, MODELLIST, **fitargs):
          model = int(train.keras_model_method)
          nepochs=MODELLIST[model]['nepochs']
          batchsize=MODELLIST[model]['batchsize']
          train.keras_model.fit(x=np.asarray(data.x), y=np.asarray(data.y), epochs=nepochs,
                                        batch_size=batchsize,
                                        **fitargs)
          return train.keras_model

def fitModelFastReweight(data, model, epochs, batchsize, sample_weights):
          model.fit(x=np.asarray(data.x), y=np.asarray(data.y), epochs=epochs, batch_size=batchsize, sample_weight=sample_weights)
          return model

def getTrainingData(train, ndata, custom=False, bin_weights=[]):
          iterations = int(round(ndata / train.train_data.batch_size))
          print("%d data points per batch * %d batches = % d total" % (train.train_data.batch_size, iterations, ndata))
          x_train = []
          y_train = []
          i = 0
          for x, y in train.train_data.generator():
                    x_train = x_train + list(x[0])
                    y_train = y_train + list(y)
                    if i == iterations:
                              break
                    else:
                              i += 1
          weights_y = [sum(Column(y_train, i)) for i in range(0, len(y_train[0]))]
          train_data = Data(length=ndata, y=y_train, x=x_train, y_weights=weights_y)
          return train_data

def getTestingData(train, reweight, ndata):
          x_test = []
          y_test = []
          i = 0
          weights = np.zeros(len(reweight))
          num = 0
          for x, y in train.train_data.generator():
                    for i in range(0, train.train_data.batch_size):
                              y_dat = y[i]
                              x_dat = x[0][i]
                              yclass = [index for index, item in enumerate(y_dat) if item > 0][0]
                              if weights[yclass] >= reweight[yclass]:
                                        pass
                              else:
                                        weights[yclass] += 1
                                        num += 1
                                        x_test.append(x_dat)
                                        y_test.append(y_dat)
                              if num == ndata:
                                        break
                    if num == ndata:
                              break

          test_data = Data(length=ndata, y=y_test, x=x_test, y_weights=weights)
          return test_data

def reweightData(weights, ndata, w):
          reweight = VecScalar(w, ndata)
          reweight = VecAdd(reweight, weights)
          if min(reweight) < 0:
                    print("There are negative values in the reweighting function, please adjust the w vector in the function reweightData()")
                    print(w)
                    print(reweight)
                    raise ValueError("Please reweight the testing data")
          else:
                    return reweight
