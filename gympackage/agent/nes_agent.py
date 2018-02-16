import gym
from gym import wrappers
import numpy as np
import time
import random
from keras import backend as k
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
import tensorflow as tf


def huber_loss(y, y_pred, delta=1.0):
    err = y - y_pred
    cond = k.abs(err) < delta
    loss = tf.where(cond, 0.5 * k.square(err), delta * (k.abs(err) - 0.5 * delta))
    return k.mean(loss)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class NN_NES_Agent:
    def __init__(self, is_action_discrete, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.isDiscrete = is_action_discrete
        self.num_action_space = action_size

        self.model = Sequential()
        self.model.add(Dense(10, activation='relu', input_dim=state_size))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huber_loss, optimizer=self.optimizer)

    def get_action(self, state, episode, network):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0 + episode)
        if self.isDiscrete:
            if epsilon <= np.random.uniform(0, 1):
                qs = softmax(network.model.predict(state)[0])
                action = np.argmax(qs)
            else:
                action = np.random.choice(range(self.num_action_space))  # ランダムに行動する
        else:
            if epsilon <= np.random.uniform(0, 1):
                qs = network.model.predict(state)[0]
                action = qs
            else:
                action = np.random.choice(range(self.num_action_space))  # ランダムに行動する

        return action
