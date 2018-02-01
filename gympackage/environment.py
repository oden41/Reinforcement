import gym
from gym import wrappers
import numpy as np
import time


class Environment:
    def __init__(self, env_name='CartPole-v0', movie_timing=(lambda ep: ep % 100 == 0)):
        env = gym.make(env_name)
        self.env = wrappers.Monitor(env, './movie/{0}'.format(env_name), force=True,
                                    video_callable=movie_timing)
        self.min_reward = env.reward_range[0]
        self.max_reward = env.reward_range[1]

        self.max_steps_episode = env.spec.max_episode_steps
        self.trials = env.spec.trials
        self.reward_threshold = env.spec.reward_threshold

        self.num_action_space = env.action_space.n
        self.num_state_space = env.observation_space.shape[0]

        self.is_complete = False
