import random as rand


class Agent:
    """エージェントクラス　テーブルやNNなどはここを継承していく"""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, episode, state):
        return rand.randint(0, self.action_space.shape[0] - 1)

    def update(self, state, action, next_state, next_action):
        pass
