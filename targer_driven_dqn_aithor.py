import random, numpy, math, gym, sys
import gym_gazebo

from keras import backend as K
from keras.models import Model

import tensorflow as tf



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
import matplotlib.image as mpimg
#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

#----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.target_model = self._createModel() 

    def create_base_network(self, input_dim):
        # Network structure must be directly changed here.
        base_model = Sequential()
        base_model.add(Convolution2D(16, 3, 3, subsample=(2,2), input_shape=(input_dim)))
        base_model.add(Activation('relu'))
        base_model.add(ZeroPadding2D((1, 1)))
        base_model.add(Convolution2D(16, 3, 3, subsample=(2,2)))
        base_model.add(Activation('relu'))
        base_model.add(Flatten())
        base_model.add(Dense(256, activation='relu'))

        return base_model
        
    def _createModel(self):
        # network definition
        base_network = self.create_base_network(stateCnt)

        input_state = Input(shape=(stateCnt))
        input_target = Input(shape=(stateCnt))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        state_branch = base_network(input_state)
        target_branch = base_network(input_target)

        merged = concatenate([state_branch, target_branch], axis=-1)

        fusion = Dense(515,activation='relu',name='loss1/fusion')(merged)

        fc1 = Dense(256,activation='relu',name='loss1/fc1')(fusion)
        fc2 = Dense(256,activation='relu',name='loss1/fc2')(fc1)

        qValue = Dense(actionCnt,activation='relu',name='qValue')(fc2)


        model = Model(input=[input_state, input_target], output=qValue)

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, t, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit([x, t], y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def getQValues(self, s, t, target=False):
        if target:
            predicted = self.target_model.predict([s,t])
        else:
            predicted = self.model.predict([s,t])
        return predicted[0]    

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def getBatch(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

    def getLength(self):
        return len(self.samples)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s, t):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.getQValues(s, t))

    def remember(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.getBatch(BATCH_SIZE)
        batchLen = len(batch)

        X_batch_state = np.empty((1,img_rows,img_cols,img_channels), dtype = np.float64)
        X_batch_target = np.empty((1,img_rows,img_cols,img_channels), dtype = np.float64)
        Y_batch = np.empty((1,self.actionCnt), dtype = np.float64)
        
        for i in range(batchLen):
            sample = batch[i]
            state = sample[0]; 
            action = sample[1]; 
            reward = sample[2]; 
            next_state = sample[3]
            target = sample[4]
            done = sample[5]

            qValue = self.brain.getQValues(state, target)
            
            targetValue = qValue
            if next_state is None:
                targetValue[action] = reward
            else:
                qValuesNextState = self.brain.getQValues(next_state, target, target=True)
                targetValue[action] = reward + GAMMA * numpy.amax(qValuesNextState)

            X_batch_state = np.append(X_batch_state, state.copy(), axis=0)
            X_batch_target = np.append(X_batch_target, target.copy(), axis=0)
            Y_batch = np.append(Y_batch, np.array([targetValue]), axis=0)

        self.brain.train(X_batch_state, X_batch_target, Y_batch, BATCH_SIZE)



class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- MAIN ----------------------------
EPISODES = 10000
START_LEARING= 1000
PROBLEM = 'GazeboCircuit2cTurtlebotCameraNnEnv-v0'
outdir = './result'
path = './result'
name_file = 'deepQlearning'

env = general_env.environment()
tmp = env.get_observation()
img_rows, img_cols, img_channels = tmp.shape[0], tmp.shape[1], tmp.shape[2]
stateCnt  = (img_rows, img_cols, img_channels)
actionCnt = 6

agent = Agent(stateCnt, actionCnt)



randomAgent = RandomAgent(actionCnt)
counter = 0
while randomAgent.memory.isFull() == False:
    s = env.get_observation()    
    s = np.array([s])
    
    ...
    ...
    target = mpimg.imread('target.png')
    target = np.array([target])
    while True:            
        counter += 1
        print counter
        a = randomAgent.act(s)
        s_, r, done = env.step(a)
        s_ = np.array([s_])

        if done: # terminal state
            s_ = None
        randomAgent.observe( (s, a, r, s_,target,done) )
        s = s_
        if done:
            break

agent.memory.samples = randomAgent.memory.samples
randomAgent = None


list_reward = []
frames = 0
for e in range(EPISODES):
    print("Episode:", e)

    #====================================================================
    state = env.get_observation()
    state = np.array([state])
    target = mpimg.imread('target.png')
    target = np.array([target])
    print "state", state.shape
    print "target", target.shape

    total_Reward = 0 

    for time in range(1000):           
        # self.env.render()
        print "step ", time
        action = agent.act(state, target)

        next_state, reward, done = env.step(action)
        next_state = np.array([next_state])


        frames += 1
        if done: # terminal state
            next_state = None

        agent.remember( (state, action, reward, next_state, target, done) )

        if frames > START_LEARING:
            agent.replay()

        state = next_state
        total_Reward += reward

        if (time == 999):
            print ("reached the end")
            done = True
        if done:
            print("Total reward:", total_Reward)
            print("Frames", frames)
            list_reward.append(total_Reward);
            break
    #====================================================================

    with open(path + "/" + name_file + ".txt", "w") as file:
        for i in list_reward:
            file.write(''.join([str(i),'\n']))

agent.brain.model.save(path + "/" + name_file + ".h5")

