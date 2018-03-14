import general_env
import numpy as np
import random
import memory
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD , Adam


class DQN:
  def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart = 128):
    """
    Parameters:
        - outputs: output size
        - memorySize: size of the memory that will store each state
        - discountFactor: the discount factor (gamma)
        - learningRate: learning rate
        - learnStart: steps to happen before for learning. Set to 128
    """
    self.output_size = outputs
    self.memory = memory.Memory(memorySize)
    self.discountFactor = discountFactor
    self.learnStart = learnStart
    self.learning_rate = learningRate

  def initNetworks(self):
    model = self.createModel()
    self.model = model
    self.targetModel = model

  def createModel(self):
    # Network structure must be directly changed here. TODO
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, subsample=(2,2),
        input_shape=(3,300,300)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(network_outputs))
    model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
    print model.summary()
    return model

  def updateTargetNetwork(self):
    self.targetModel.set_weights(self.model.get_weights())

  def getQValues(self, state):
    predicted = self.model.predict(state)
    return predicted[0]

  def getTargetQValues(self, state):
    predicted = self.targetModel.predict(state)
    return predicted[0]

  def getMaxQ(self, qValues):
    return np.max(qValues)

  def getMaxIndex(self, qValues):
    return np.argmax(qValues)

  def selectAction(self, qValues, explorationRate):
    rand = random.random()
    if rand < explorationRate :
      action = np.random.randint(0, self.output_size)
    else :
      action = self.getMaxIndex(qValues)
    return action

  def calculateTarget(self, qValuesNewState, reward, isFinal):
    """
    target = reward(s,a) + gamma * max(Q(s')tch_outs = f(ins_batch)
  File "/home/hmi/.local/lib/python2.7/site-packages/keras/backend/tensorflow_backend.py", line 2355, in __call__
    session = get_session()
  File "/home/hmi/.local/lib/python2.7/site-packages/keras/backend/tensorflow_backend.py", line 169, in get_session

    """
    if isFinal:
      return reward
    else :
      return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

  def addMemory(self, state, action, reward, newState, isFinal):
    self.memory.addMemory(state, action, reward, newState, isFinal)

  def learnOnLastState(self):
    if self.memory.getCurrentSize() >= 1:
      return self.memory.getMemory(self.memory.getCurrentSize() - 1)

  def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
    # Do not learn until we've got self.learnStart samples
    if self.memory.getCurrentSize() > self.learnStart:
      # learn in batches of 128
      miniBatch = self.memory.getMiniBatch(miniBatchSize)
      X_batch = np.empty((1,3,300,300), dtype = np.float64)
      Y_batch = np.empty((1,self.output_size), dtype = np.float64)
      for sample in miniBatch:
        isFinal = sample['isFinal']
        state = sample['state']
        action = sample['action']
        reward = sample['reward']
        newState = sample['newState']

        qValues = self.getQValues(state)
        if useTargetNetwork:
          qValuesNewState = self.getTargetQValues(newState)
        else:
          qValuesNewState = self.getQValues(newState)
        targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
        X_batch = np.append(X_batch, state.copy(), axis=0)
        Y_sample = qValues.copy()
        Y_sample[action] = targetValue
        Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
        if isFinal:
          X_batch = np.append(X_batch, newState.copy(), axis=0)
          Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
      self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)


if __name__ == "__main__" : 

  my_env = general_env.environment()
  #TODO inital parameter
  epochs = 100000
  steps = 5000
  minibatch_size = 32
  learningRate = 1e-3#1e6
  discountFactor = 0.95
  network_outputs = my_env.action_range
  memorySize = 100000
  learnStart = 1000 # timesteps to observe before training
  EXPLORE = memorySize # frames over which to anneal epsilon
  INITIAL_EPSILON = 1 # starting value of epsilon
  FINAL_EPSILON = 0.01 # final value of epsilon
  explorationRate = INITIAL_EPSILON
  stepCounter = 0

  deepQ = DQN(network_outputs, memorySize, discountFactor, learningRate, learnStart)
  deepQ.initNetworks()
  #TODO training

  for epoch in range(epochs):
    my_env.env_reset()
    observation = my_env.get_observation()
    cumulated_reward = 0


    for t in xrange(steps):
      observation = np.expand_dims(observation.reshape(3,300,300), axis = 0)
      qValues = deepQ.getQValues(observation)

      action = deepQ.selectAction(qValues, explorationRate)
      newObservation, reward, done = my_env.step(action)
      newObservation = np.expand_dims(newObservation.reshape(3,300,300), axis = 0)
      cumulated_reward += reward
      deepQ.addMemory(observation, action, reward, newObservation, done)
      deepQ.updateTargetNetwork()
      observation = newObservation

      #We reduced the epsilon gradually
      if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
          explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

      if stepCounter == learnStart:
          print("Starting learning")

      if stepCounter >= learnStart:
          deepQ.learnOnMiniBatch(minibatch_size, False)

      if (t == steps-1):
          print ("reached the end")
          done = True

      if done:
        print ("Epoch"+ str(epoch) + "step: " + str(t) + "reward: " + str(cumulated_reward))
        break

      stepCounter += 1



