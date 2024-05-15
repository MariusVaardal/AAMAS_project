import numpy as np

class RandomAgent:
    def __init__(self, name: str):
        self.name = name
        self.observation = None

    def see(self, observation):
        self.observation = observation

    def get_action(self) -> int:
        return np.random.randint(5)