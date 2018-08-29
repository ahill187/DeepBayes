from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore.DataCollection_AH import DataCollection
import numpy as np
from pdb import set_trace

import imp
from Losses import *
global_layers_list = {}
from Metrics import *
global_metrics_list = {}
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)
from RecoilModels import *
from RecoilModels_AH import denseDNN
from keras.losses import categorical_crossentropy
from Matrix import *

class training_base(object):
          def __init__(self, splittrainandtest=0.85, useweights=False, testrun=False, resumeSilently=False,
		        renewtokens=True, collection_class=DataCollection, parser=None, pythonconsole=False, treefiles=""):
                    import sys
                    scriptname=sys.argv[0] # this returns the name of the script calling training_base()

                    # Getting input from bash console
                    if parser is None: parser = ArgumentParser('Run the training')
                    parser.add_argument('inputDataCollection')
                    parser.add_argument('outputDir')
                    parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class', metavar='OPT', default=None)
                    parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
                    parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)

                    args = parser.parse_args()
                    # If running in python console:
                    if pythonconsole:
                              f = open(treefiles)
                              arguments = f.readlines()
                              args.inputDataCollection = arguments[0][0:-1]
                              args.outputDir = arguments[1][0:-1]
                              args.modelMethod = arguments[2][0:-1]
                              scriptname = '/home/ahill/DeepLearning/CMSSW_10_2_0_pre5/src/DeepML/Train/test_TrainData_Recoil.py'
                    self.args = args
                    import os
                    import matplotlib
                    #if no X11 use below
                    matplotlib.use('Agg')
                    if args.gpu<0:
                              import imp
                              try:
                                        imp.find_module('setGPU')
                                        import setGPU
                              except ImportError:
                                        found = False
                    else:
                              os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                              os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                              print('running on GPU '+str(args.gpu))
                    print("Checkpoint 4")
                    if args.gpufraction>0 and args.gpufraction<1:
                              import sys
                              import tensorflow as tf
                              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
                              sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                              import keras
                              from keras import backend as K
                              K.set_session(sess)
                              print('using gpu memory fraction: '+str(args.gpufraction))
                    print("Checkpoint 5")
                    import keras

                    self.keras_inputs=[]
                    self.keras_inputsshapes=[]
                    self.keras_model=None
                    self.keras_model_method=args.modelMethod
                    self.train_data=None
                    self.val_data=None
                    self.startlearningrate=None
                    self.optimizer=None
                    self.trainedepoches=0
                    self.compiled=False
                    self.checkpointcounter=0
                    self.renewtokens=renewtokens
                    self.callbacks=None
                    self.inputData = os.path.abspath(args.inputDataCollection) \
												 if ',' not in args.inputDataCollection else \
														[os.path.abspath(i) for i in args.inputDataCollection.split(',')]
                    self.outputDir=args.outputDir
                    # create output dir
                    print("Checkpoint 6")
                    isNewTraining=True
                    if os.path.isdir(self.outputDir):
                              if not resumeSilently:
                                        var = raw_input('output dir exists. To recover a training, please type "yes"\n')
                              if not var == 'yes':
                                        raise Exception('output directory must not exist yet')
                              isNewTraining=False
                    else:
                              os.mkdir(self.outputDir)
                    print("Checkpoint 7")
                    self.outputDir = os.path.abspath(self.outputDir)
                    self.outputDir+='/'
                    print("Checkpoint 8")
                    #copy configuration to output dir
                    shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))

                    self.train_data = collection_class()
                    self.train_data.readFromFile(self.inputData)
                    self.train_data.useweights=useweights
                    print("Checkpoint 9")
                    if testrun:
                              self.train_data.split(0.02)
                              self.val_data=self.train_data
                    else:
                              self.val_data=self.train_data.split(splittrainandtest)

                    print("Checkpoint 10")

                    shapes=self.train_data.getInputShapes()
                    self.train_data.maxFilesOpen=-1

                    self.keras_inputs=[]
                    self.keras_inputsshapes=[]

                    print(shapes)

                    for s in shapes:
                              self.keras_inputs.append(keras.layers.Input(shape=s))
                              self.keras_inputsshapes.append(s)

                    if not isNewTraining:
                              kfile = self.outputDir+'KERAS_check_model_last.h5' \
							 if os.path.isfile(self.outputDir+'KERAS_check_model_last.h5') else \
							 self.outputDir+'KERAS_model.h5'
                              if not os.path.isfile(kfile):
                                        print('you cannot resume a training that did not train for at least one epoch.\nplease start a new training.')
                                        exit()
                              self.loadModel(kfile)
                              self.trainedepoches=sum(1 for line in open(self.outputDir+'losses.log'))


          def __del__(self):
                    if hasattr(self, 'train_data'):
                              del self.train_data
                              del self.val_data

          def modelSet(self):
                    return not self.keras_model==None

          def setModel(self,model,**modelargs):
                    if len(self.keras_inputs)<1:
                              raise Exception('setup data first')
                    self.keras_model=model(self.keras_inputs,
                               self.train_data.getNClassificationTargets(),
                               self.train_data.getNRegressionTargets(),
                               **modelargs)
                    if not self.keras_model:
                              raise Exception('Setting model not successful')

          def defineCustomPredictionLabels(self, labels):
                    self.train_data.defineCustomPredictionLabels(labels)
                    self.val_data.defineCustomPredictionLabels(labels)

          def saveCheckPoint(self,addstring=''):
                    self.checkpointcounter=self.checkpointcounter+1
                    self.saveModel("KERAS_model_checkpoint_"+str(self.checkpointcounter)+"_"+addstring +".h5")

          def loadModel(self,filename):
                    print(filename)
                    from keras.models import load_model
                    self.keras_model=load_model(filename, custom_objects=custom_objects_list)
                    self.optimizer=self.keras_model.optimizer
                    self.compiled=True

          def compileModel(self, learningrate, clipnorm=None, **compileargs):
                    if not self.keras_model:
                              raise Exception('set model first')

                    from keras.optimizers import Adam
                    self.startlearningrate=learningrate
                    if clipnorm:
                              self.optimizer = Adam(lr=self.startlearningrate,clipnorm=clipnorm)
                    else:
                              self.optimizer = Adam(lr=self.startlearningrate)
                    self.keras_model.compile(optimizer=self.optimizer,**compileargs)
                    self.compiled=True

          def compileModelWithCustomOptimizer(self, customOptimizer, **compileargs):
                    if not self.keras_model:
                              raise Exception('set model first')
                    self.optimizer = customOptimizer
                    self.keras_model.compile(optimizer=self.optimizer,**compileargs)
                    self.compiled=True

          def saveModel(self,outfile):
                    self.keras_model.save(self.outputDir+outfile)
                    import tensorflow as tf
                    import keras.backend as K
                    tfsession=K.get_session()
                    saver = tf.train.Saver()
                    tfoutpath=self.outputDir+outfile+'_tfsession/tf'
                    import os
                    os.system('rm -rf '+tfoutpath)
                    os.system('mkdir -p '+tfoutpath)
                    saver.save(tfsession, tfoutpath)


          #import h5py
          #f = h5py.File(self.outputDir+outfile, 'r+')
          #del f['optimizer_weights']
          #f.close()

          def trainModel(self, nepochs, batchsize, stop_patience=-1,  lr_factor=0.5, lr_patience=-1,  lr_epsilon=0.003,
                   lr_cooldown=6, lr_minimum=0.000001, maxqsize=5, checkperiod=10, additional_plots=None, **trainargs):

                    # check a few things, e.g. output dimensions etc.
                    # need implementation, but probably TF update SWAPNEEL
                    customtarget=self.train_data.getCustomPredictionLabels()
                    if customtarget:
                              pass
                    # work on self.model.outputs
                    # check here if the output dimension of the model fits the custom labels

                    # write only after the output classes have been added
                    self.train_data.writeToFile(self.outputDir+'trainsamples.dc')
                    self.val_data.writeToFile(self.outputDir+'valsamples.dc')

                    #make sure tokens don't expire
                    #from .tokenTools import checkTokens, renew_token_process
                    from DeepJetCore.training import tokenTools
                    from thread import start_new_thread

                    if self.renewtokens:
                              print('starting afs backgrounder')
                              checkTokens()
                              start_new_thread(renew_token_process,())

                    self.train_data.setBatchSize(batchsize)
                    self.val_data.setBatchSize(batchsize)

                    averagesamplesperfile=self.train_data.getAvEntriesPerFile()
                    samplespreread=maxqsize*batchsize
                    nfilespre=max(int(samplespreread/averagesamplesperfile),2)
                    nfilespre+=1
                    nfilespre=min(nfilespre, len(self.train_data.samples)-1)
                    #if nfilespre>15:nfilespre=15
                    print('best pre read: '+str(nfilespre)+'  a: '+str(int(averagesamplesperfile)))
                    print('total sample size: '+str(self.train_data.nsamples))
                    #exit()

                    if self.train_data.maxFilesOpen<0 or True:
                              self.train_data.maxFilesOpen=nfilespre
                              self.val_data.maxFilesOpen=min(int(nfilespre/2),1)

                    #self.keras_model.save(self.outputDir+'KERAS_check_last_model.h5')
                    print('setting up callbacks')
                    #from .DeepJet_callbacks import DeepJet_callbacks
                    from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks
                    #import DeepJetCore.training.DeepJet_callbacks

                    self.callbacks=DeepJet_callbacks(self.keras_model,
                                    stop_patience=stop_patience,
                                    lr_factor=lr_factor,
                                    lr_patience=lr_patience,
                                    lr_epsilon=lr_epsilon,
                                    lr_cooldown=lr_cooldown,
                                    lr_minimum=lr_minimum,
                                    outputDir=self.outputDir,
                                    checkperiod=checkperiod,
                                    checkperiodoffset=self.trainedepoches,
                                    additional_plots=additional_plots)

                    print('starting training')
                    self.keras_model.fit_generator(self.train_data.generator() ,
                            steps_per_epoch=self.train_data.getNBatchesPerEpoch(),
                            epochs=nepochs-self.trainedepoches,
                            callbacks=self.callbacks.callbacks,
                            validation_data=self.val_data.generator(),
                            validation_steps=self.val_data.getNBatchesPerEpoch(), #)#,
                            max_q_size=maxqsize,**trainargs)

                    self.trainedepoches=nepochs
                    self.saveModel("KERAS_model.h5")

                    import copy
                    #reset all file reads etc
                    tmpdc=copy.deepcopy(self.train_data)
                    del self.train_data
                    self.train_data=tmpdc

                    return self.keras_model, self.callbacks.history


MODELLIST= {
          # SCALE
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

          # DIRECTION
          100:  { "method":meanDNN,                'arch':'32x16x4',           "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":huber,          "learningrate":0.0001,  "nepochs":30, "batchsize":1024 },
          110:  { "method":meanpquantilesDNN,      'arch':':32x16x4',          "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":ahuber_q,       "learningrate":0.0001,  "nepochs":30, "batchsize":1024 },
          150:  { "method":semiParamDNN,           'arch':':32x16x4',          "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
          151:  { "method":semiParamDNN,           'arch':'32:16x4',           "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
          152:  { "method":semiParamDNN,           'arch':'32:32x16x4',        "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
          153:  { "method":semiParamDNN,           'arch':'64x32:32x16x4',     "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },
          154:  { "method":semiParamDNN,           'arch':'128x64x32:32x16x4', "dropout":None, "batchNorm": True,  "splitInputs": False, 'pfix':'_e2', "loss":global_loss_list['gd_offset_loss'], "learningrate":0.0001,  "nepochs":30, "batchsize":1024  },

          # DEEPBAYES
          200: { "method":denseDNN, 'arch':'30x30',           "dropout":0.5, "batchNorm": False, "splitInputs": False,  'pfix':'', "loss":categorical_crossentropy, "learningrate":0.0001,  "nepochs":500, "batchsize":1024, 'reg':0, 'bins':20 },
}


train=training_base(testrun=False,renewtokens=False, pythonconsole=True, treefiles='/home/ahill/output_directory/regress_results/train/treefiles.txt')
nepochs=20
batchsize=1024
model=int(train.keras_model_method)

nepochs=MODELLIST[model]['nepochs']
batchsize=MODELLIST[model]['batchsize']
train.setModel( model=MODELLIST[model]['method'],
                    arch=MODELLIST[model]['arch'],
                    dropoutRate=MODELLIST[model]['dropout'],
                    batchNorm=MODELLIST[model]['batchNorm'],
                    splitInputs=MODELLIST[model]['splitInputs'],
                    pfix=MODELLIST[model]['pfix'],
                    reg=MODELLIST[model]['reg'],
                    bins=MODELLIST[model]['bins'])

metricsList=['acc', 'mse', 'mae']
predictionLabels=['mu']
predictionLabels = [x+MODELLIST[model]['pfix'] for x in predictionLabels]
predictionLabels = np.arange(10)
predictionLabels = [str(x) for x in predictionLabels]
train.defineCustomPredictionLabels(predictionLabels)

train.compileModel(learningrate=MODELLIST[model]['learningrate'],
                       loss=MODELLIST[model]['loss'],
                       metrics=metricsList)
print(train.keras_model.summary())

nepochs=1
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

#  i=0
# for check in train.train_data.generator():
#           a = check
#           i = i+1
#           if i==2:
#                     break

