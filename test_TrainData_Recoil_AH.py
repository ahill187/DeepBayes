from DeepJetCore.training.training_base import training_base
import sys
sys.path.append('/home/ahill/DeepLearning/CMSSW_10_2_0_pre5/src/DeepML/modules/')
sys.path.append('/home/ahill/DeepLearning/CMSSW_10_2_0_pre5/src/DeepML/Train')
from RecoilModels import *
from Losses import *
from Metrics import *

import copy
print("Checkpoint")
#a list of methods (name, batchNorm, dropoutRate, loss)
MODELLIST= {
    #
    # SCALE
    #
    #non-parametric: mean
    0:   { "method":meanDNN, 'arch':'32x16x4',           "dropout":0.2, "batchNorm": True, "splitInputs": False,  'pfix':'', "loss":ahuber, "learningrate":0.0001,  "nepochs":50, "batchsize":1024 },

    #non-parametric: mean+quantiles
    10:   { "method":meanpquantilesDNN, 'arch':'32x16x4',          "dropout":0.2, "batchNorm": True, "splitInputs": False, 'pfix':'', "loss":ahuber_q, "learningrate":0.0001, "nepochs":100, "batchsize":1024 },
    11:   { "method":meanpquantilesDNN, 'arch':'32x8x32:16x4',     "dropout":0.2, "batchNorm": True, "splitInputs": False, 'pfix':'', "loss":ahuber_q, "learningrate":0.0001, "nepochs":100, "batchsize":1024 },

    #semi-parametric
    50:  { "method":semiParamDNN,       'arch':':32x16x4',         "dropout":None, "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },
    51:  { "method":semiParamDNN,       'arch':':32x16x4',         "dropout":0.2,  "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },
    60:  { "method":semiParamDNN,       'arch':':64x32x16',        "dropout":0.2,  "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },
    70:  { "method":semiParamDNN,       'arch':':128x64x32',       "dropout":0.2,  "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },
    80:  { "method":semiParamDNN,       'arch':'64x32:32x16x4',    "dropout":0.2,  "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },
    90:  { "method":semiParamDNN,       'arch':'32x8x32:32x16x4',  "dropout":0.2,  "batchNorm": True,  "splitInputs": False,  'pfix':'', "loss":global_loss_list['gd_loss'],  "learningrate":0.0001,  "nepochs":150,  "batchsize":1024 },

    #
    # DIRECTION
    #
    100:  { "method":meanDNN,                'arch':'32x16x4',           "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":huber,          "learningrate":0.0001,  "nepochs":30, "batchsize":1024 },
    110:  { "method":meanpquantilesDNN,      'arch':':32x16x4',          "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":ahuber_q,       "learningrate":0.0001,  "nepochs":30, "batchsize":1024 },
    150:  { "method":semiParamDNN,           'arch':':32x16x4',          "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
    151:  { "method":semiParamDNN,           'arch':'32:16x4',           "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
    152:  { "method":semiParamDNN,           'arch':'32:32x16x4',        "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
    153:  { "method":semiParamDNN,           'arch':'64x32:32x16x4',     "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
    154:  { "method":semiParamDNN,           'arch':'128x64x32:32x16x4', "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
    }

#start a training base class (also does the parsing)
#if running locally for >24h you may want to set renewtokens=True

def GetTrees():
          train=training_base(testrun=False,renewtokens=False, pythonconsole=True, treefiles='/home/ahill/DeepLearning/CMSSW_10_2_0_pre5/src/DeepML/regress_results/train/treefiles.txt')
          print(train.inputData)
          print(train.outputDir)
          print(train.args)
          return train

train.train_data.generator()


nepochs=20
batchsize=1024
if  not train.modelSet():
    model=int(train.keras_model_method)
    if not model in MODELLIST:
        raise ValueError('Unknown model %d'%model)
    print 'Setting model',model
    print MODELLIST[model]

    nepochs=MODELLIST[model]['nepochs']
    batchsize=MODELLIST[model]['batchsize']
    train.setModel( model=MODELLIST[model]['method'],
                    arch=MODELLIST[model]['arch'],
                    dropoutRate=MODELLIST[model]['dropout'],
                    batchNorm=MODELLIST[model]['batchNorm'],
                    splitInputs=MODELLIST[model]['splitInputs'],
                    pfix=MODELLIST[model]['pfix'])

    predictionLabels=['mu']
    metricsList=['mae','mse']
    if MODELLIST[model]['loss']==ahuber_q:
        predictionLabels=['mu','qm','qp']
    if MODELLIST[model]['loss']==global_loss_list['gd_loss_gauss_fixedtails']:
        predictionLabels=['mu','sigma']
    if  MODELLIST[model]['loss'] in [global_loss_list['gd_loss'],global_loss_list['gd_loss_gauss']]:
        predictionLabels=['mu','sigma','a1','a2']
    if  MODELLIST[model]['loss'] in [global_loss_list['gd_offset_loss'],global_loss_list['gd_offset_loss_gauss']]:
        predictionLabels=['mu','sigma','a1','a2','n']
    if 'e2' in MODELLIST[model]['pfix']:
        metricsList += [dir_metric]
    else:
        metricsList += [scale_metric]
    predictionLabels = [x+MODELLIST[model]['pfix'] for x in predictionLabels]
    train.defineCustomPredictionLabels(predictionLabels)

    train.compileModel(learningrate=MODELLIST[model]['learningrate'],
                       loss=MODELLIST[model]['loss'],
                       metrics=metricsList)
    print(train.keras_model.summary())


model,history = train.trainModel(nepochs=nepochs,
                                 batchsize=batchsize,
                                 stop_patience=300, #stop after N-epochs if validation loss increases
                                 lr_factor=0.5,     #adapt learning rate if validation loss flattens
                                 lr_patience=15,
                                 lr_epsilon=0.0001,
                                 lr_cooldown=2,
                                 lr_minimum=0.0001,
                                 maxqsize=100       #play if file system is unstable
                             )
