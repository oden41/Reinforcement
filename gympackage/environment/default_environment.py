import gym
from gym import wrappers
from gym.spaces import Discrete
import numpy as np


class Environment:
    def __init__(self, env_name='CartPole-v0', movie_timing=(lambda ep: ep % 100 == 0)):
        env = gym.make(env_name)
        self.env = wrappers.Monitor(env, './movie/{0}'.format(env_name), force=True, video_callable=movie_timing)
        self.min_reward = env.reward_range[0]
        self.max_reward = env.reward_range[1]

        self.max_steps_episode = env.spec.max_episode_steps
        self.trials = env.spec.trials
        self.reward_threshold = env.spec.reward_threshold

        self.__reward_vec = np.array([-10000] * self.trials)
        self.sum_reward = 0

        self.action_space = env.action_space
        self.state_space = env.observation_space
        self.is_action_space_discrete = isinstance(self.action_space, Discrete)
        self.is_state_space_discrete = isinstance(self.state_space, Discrete)
        self.num_state_space = env.observation_space.shape[0]

    def initialize(self):
        """環境初期化　状態を返す"""
        self.__reward_vec = np.array([-10000] * self.trials)
        self.sum_reward = 0
        return self.env.reset()

    def reset(self):
        """エピソード初期化　状態を返す"""
        self.sum_reward = 0
        return self.env.reset()

    def next_step(self, action):
        """行動を取り，その結果の観測一覧を返す"""
        obs = self.env.step(action)
        obs = Observation(obs)
        self.sum_reward += obs.reward
        return obs

    def get_reward_ave(self):
        return self.__reward_vec.mean()

    def end(self):
        self.__reward_vec = np.hstack((self.__reward_vec[1:], self.sum_reward))

    def is_clear(self):
        return self.get_reward_ave() > self.reward_threshold


class Observation:
    def __init__(self, obs):
        """info (dict): information useful for debugging. """
        self.state = obs[0]
        self.reward = obs[1]
        self.done = obs[2]
        self.info = obs[3]
