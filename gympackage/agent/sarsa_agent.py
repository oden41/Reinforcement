import random as rand
import numpy as np


class SARSA_Agent:
    """SARSAによる更新エージェントクラス"""

    def __init__(self, action_space, num_d):
        super(action_space)
        self.q_table = np.random.uniform(low=-1, high=1, size=(num_d ** 4, action_space.n))

    def get_action(self, episode, state):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[state])
        else:
            next_action = np.random.choice([0, 1])
        return next_action

    def update(self, state, action, reward, next_state, next_action):
        gamma = 0.99
        alpha = 0.5
        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (
            reward + gamma * self.q_table[next_state, next_action])
