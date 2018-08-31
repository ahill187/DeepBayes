from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras import backend as K
import keras.regularizers as regularizers


def meanDNN(Inputs, nclasses, nregclasses, dropoutRate=None, batchNorm=True, splitInputs=False, pfix='',
            arch='30x15x5'):
          """mean regression with a DNN"""

          nodes = arch.split('x')
          arch = [(i + 1, int(nodes[i]), 'lrelu') for i in xrange(0, len(nodes))]

          if batchNorm:
                    x = BatchNormalization(name='bn_0')(Inputs[0])
          else:
                    x = Inputs[0]
          for ilayer, isize, iact in arch:
                    x = Dense(isize,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='dense_%d' % ilayer)(x)
                    if batchNorm:
                              x = BatchNormalization(name='bn_%d' % ilayer)(x)
                    if dropoutRate:
                              x = Dropout(dropoutRate)(x)
                    if iact == 'lrelu':
                              x = LeakyReLU(0.2, name='lrelu_act_%d' % ilayer)(x)
                    if iact == 'prelu':
                              x = PReLU(name='prelu_act_%d' % ilayer)(x)

          if splitInputs:
                    x = Dense(1,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='pre_mu_1')(x)
                    x = Add(name='add_splitinput')([x, Inputs[1]])
                    x = Dense(2,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='pre_mu_2')(x)

          x = Dense(1, use_bias=True, name='mu' + pfix)(x)

          return Model(inputs=Inputs, outputs=[x])

def meanpquantilesDNN(Inputs, nclasses, nregclasses, dropoutRate=None, batchNorm=True, splitInputs=False, pfix='',
                      arch=':32x16x4'):
          """model for the regression of the recoil scale"""

          x = BatchNormalization(name='bn_0')(Inputs[0]) if batchNorm else Inputs[0]

          # parse the architecture
          nodesCommon, nodesInd = arch.split(':')

          # common nodes
          archCommon = []
          if len(nodesCommon) > 0:
                    nodesCommon = nodesCommon.split('x')
                    archCommon = [(i + 1, int(nodesCommon[i]), 'lrelu') for i in xrange(0, len(nodesCommon))]
          for ilayer, isize, iact in archCommon:
                    x = Dense(isize,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='dense_%d' % (ilayer))(x)
                    if batchNorm:
                              x = BatchNormalization(name='bn_%d' % (ilayer))(x)
                    if dropoutRate:
                              x = Dropout(dropoutRate)(x)
                    if iact == 'lrelu':
                              x = LeakyReLU(0.2)(x)

          # individual nodes for the 3 parameters
          nodesInd = nodesInd.split('x')
          archInd = [(i + 1, int(nodesInd[i]), 'lrelu') for i in xrange(0, len(nodesInd))]
          param_nn = {}
          for p in ['mu', 'qm', 'qp']:

                    # y = x if p=='mu' else  Add(name='add_%s'%p)( [x,param_nn['mu']] )
                    y = x
                    param_nn[p] = BatchNormalization(name='bn_%s_0' % p)(y) if batchNorm else y
                    for ilayer, isize, iact in archInd:
                              param_nn[p] = Dense(isize,
                                                  kernel_initializer='glorot_normal',
                                                  bias_initializer='glorot_uniform',
                                                  name='dense_%s_%d' % (p, ilayer))(param_nn[p])
                              if batchNorm:
                                        param_nn[p] = BatchNormalization(name='bn_%s_%d' % (p, ilayer))(param_nn[p])
                              if dropoutRate:
                                        param_nn[p] = Dropout(dropoutRate)(param_nn[p])
                              if iact == 'lrelu':
                                        param_nn[p] = LeakyReLU(0.2)(param_nn[p])

                    # final layer for 1 parameter estimation
                    param_nn[p] = Dense(1,
                                        kernel_initializer='glorot_normal',
                                        bias_initializer='glorot_uniform',
                                        activation='linear',
                                        name=p)(param_nn[p])

          # build the final model
          output_global = concatenate([param_nn[p] for p in param_nn])

          # build the final model
          return Model(inputs=Inputs, outputs=[output_global])


def semiParamDNN(Inputs, nclasses, nregclasses, dropoutRate=None, batchNorm=True, splitInputs=False, pfix='',
                 arch=':32x16x4'):
          """model for the regression of the recoil and direction scale"""

          p2fit = ['mu', 'sigma', 'a1', 'a2']
          if nregclasses == 5: p2fit += ['offset']
          if nregclasses == 2: p2fit = ['mu', 'sigma']

          x = BatchNormalization(name='bn_0')(Inputs[0]) if batchNorm else Inputs[0]

          # parse the architecture
          nodesCommon, nodesInd = arch.split(':')

          # common nodes
          archCommon = []
          if len(nodesCommon) > 0:
                    nodesCommon = nodesCommon.split('x')
                    archCommon = [(i + 1, int(nodesCommon[i]), 'lrelu') for i in xrange(0, len(nodesCommon))]
          for ilayer, isize, iact in archCommon:
                    x = Dense(isize,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='dense_%d' % (ilayer))(x)
                    if batchNorm:
                              x = BatchNormalization(name='bn_%d' % (ilayer))(x)
                    if dropoutRate:
                              x = Dropout(dropoutRate)(x)
                    if iact == 'lrelu':
                              x = LeakyReLU(0.2)(x)

          # individual nodes for the parameters
          nodesInd = nodesInd.split('x')
          archInd = [(i + 1, int(nodesInd[i]), 'lrelu') for i in xrange(0, len(nodesInd))]
          param_nn = {}
          for p in p2fit:
                    if batchNorm:
                              param_nn[p] = BatchNormalization(name='bn_%s_0' % p)(x)
                    else:
                              param_nn[p] = x

                    for ilayer, isize, iact in archInd:
                              param_nn[p] = Dense(isize,
                                                  kernel_initializer='glorot_normal',
                                                  bias_initializer='glorot_uniform',
                                                  name='dense_%s_%d' % (p, ilayer))(param_nn[p])
                              if batchNorm:
                                        param_nn[p] = BatchNormalization(name='bn_%s_%d' % (p, ilayer))(param_nn[p])
                              if dropoutRate:
                                        param_nn[p] = Dropout(dropoutRate)(param_nn[p])
                              if iact == 'lrelu':
                                        param_nn[p] = LeakyReLU(0.2)(param_nn[p])

                    # final layer
                    param_nn[p] = Dense(1,
                                        kernel_initializer='glorot_normal',
                                        bias_initializer='glorot_uniform',
                                        activation='linear',
                                        name=p)(param_nn[p])

          # build the final model
          output_global = concatenate([param_nn[p] for p in p2fit])
          return Model(inputs=Inputs, outputs=[output_global])

def denseDNN(Inputs, nclasses, nregclasses, dropoutRate=None, batchNorm=False, splitInputs=False, pfix='', arch='30x30', reg=0, bins=20):

          nodes = arch.split('x')
          arch = [(i + 1, int(nodes[i]), 'None') for i in xrange(0, len(nodes))]

          if batchNorm:
                    x = BatchNormalization(name='bn_0')(Inputs[0])
          else:
                    x = Inputs[0]
          for ilayer, isize, iact in arch:
                    x = Dense(isize, activation='relu', name='dense_%d' % ilayer)(x)
                    if batchNorm:
                              x = BatchNormalization(name='bn_%d' % ilayer)(x)
                    if dropoutRate:
                              x = Dropout(dropoutRate)(x)
                    if iact == 'lrelu':
                              x = LeakyReLU(0.2, name='lrelu_act_%d' % ilayer)(x)
                    if iact == 'prelu':
                              x = PReLU(name='prelu_act_%d' % ilayer)(x)

          if splitInputs:
                    x = Dense(1, kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='pre_mu_1')(x)
                    x = Add(name='add_splitinput')([x, Inputs[1]])
                    x = Dense(2,
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_uniform',
                              name='pre_mu_2')(x)

          x = Dense(bins, activation='softmax', kernel_regularizer=regularizers.l2(reg),  name='mu' + pfix)(x)
          return Model(inputs=Inputs, outputs=[x])
