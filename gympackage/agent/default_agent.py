import random as rand


class Agent:
    """エージェントクラス　テーブルやNNなどはここを継承していく"""

    def __init__(self, num_action):
        self.num_action = num_action

    def get_action(self, state):
        return rand.randint(0, self.num_action - 1)

    def update(self, state, action, next_state, next_action):
        pass
