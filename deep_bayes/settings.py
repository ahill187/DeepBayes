from dnn_models import denseDNN
from keras.losses import categorical_crossentropy

TESTLIST = {
          'evenbins': [0.025, 0.01, 0.063, 0.05, -0.01, -0.01, -0.01,-0.05,-0.05,-0.001,-0.001,-0.001,-0.005,0,0,0, 0, -0.01, 0, 0],
          'evenbins2': [0.02, 0.01, 0.05, 0.03, -0.01, -0.01, -0.01,-0.031,-0.031,-0.001,-0.001,-0.001,-0.005,0,0,0, 0, -0.01, 0, 0],
          'quantilebins1': [0.025, 0.02, 0.002, -0.01, -0.01, -0.01, -0.01,-0.0025,-0.0015,-0.001,-0.001,-0.001,0,0,0,0, 0, 0, 0, 0],
          'quantilebins2':[-0.005, 0, 0, 0, 0, 0.0043, 0.0043,0.025, 0.01, -0.01, -0.01, -0.01,-0.005,-0.0001,-0.001,-0.001, -0.0005, -0.0005, -0.0005, 0]}

MODELLIST= {

          # DEEPBAYES
          200: { "method":denseDNN, 'arch':'30x30',   "dropout":0.5, "batchNorm": False, "splitInputs": False,  'pfix':'', "loss":categorical_crossentropy, "learningrate":0.00001,  "nepochs":200, "batchsize":1024, 'reg':0, 'bins':20},
          201: { "method":denseDNN, 'arch':'100x100x30x100',   "dropout":0.5, "batchNorm": False, "splitInputs": False,  'pfix':'', "loss":categorical_crossentropy, "learningrate":0.00001,  "nepochs":200, "batchsize":1024, 'reg':0, 'bins':20}
}

DEFAULTLIST = {'folder':'Plots', 'epochs':25, 'epochs2':25, 'fast':False, 'ndata':10000, 'n_train':1000000,'bins':20, 'custom':True,
                                        'test_weights':'evenbins2', 'damping_constant':1, 'iterations':200}

BIN = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 10, 20]

ADJUST = {'binerror':False, 'damping':False, 'retrain':False, 'binerror2':False}