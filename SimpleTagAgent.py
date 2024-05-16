from abc import ABC, abstractmethod
import numpy as np

# This is an abstract class
class SimpleTagAgent(ABC):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        self.name = name
        self.num_adversaries = num_adversaries
        self.num_landmarks = num_landmarks
        self.agents = [f'adversary_{i}' for i in range(num_adversaries)] + ['agent_0'] 
        self.target_vec_divisor = 1
        self.action_vecs = action_vecs = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])
        self.observations = None
        self.observed_agent_positions = {}
    
    @abstractmethod
    def get_action(self) -> list:
        pass

    @abstractmethod
    def update_observed_agent_positions(self):
        pass

