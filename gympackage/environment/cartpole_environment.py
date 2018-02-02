from gympackage.environment.default_environment import Environment
import numpy as np


class CartPoleV0(Environment):
    def __init__(self, timing=(lambda ep: ep % 100 == 0)):
        super().__init__('CartPole-v0', timing)
        self.max_state_eval = np.array([2.4, 5.0, 0.41, 5.0])
        self.min_state_eval = np.array([-2.4, -5.0, -0.41, -5.0])
