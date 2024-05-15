import numpy as np

#class not used as for now
class RandomAgent:
    def __init__(self, name: str):
        self.name = name
        self.observation = None

    def see(self, observation):
        self.observation = observation

    def get_action(self) -> int:
        return np.random.randint(5)

#function to be used in simple_tag.py
def get_random_action(this_agent, observations):
    return np.random.randint(5)
