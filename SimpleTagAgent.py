from abc import ABC, abstractmethod
import numpy as np

# This is an abstract class
class SimpleTagAgent(ABC):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        self.name = name
        self.num_adversaries = num_adversaries
        self.num_landmarks = num_landmarks
        self.agents = [f'adversary_{i}' for i in range(num_adversaries)] + ['agent_0'] 
        self.observations = None
        self.observed_agent_positions = {}
        self.action_vectors = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    
    @abstractmethod
    def get_action(self) -> list:
        pass

    def see(self, obs):
        self.observation = obs

    def update_observed_agent_positions(self):
        base_index = 4 + 2 * self.num_landmarks
        self_pos = np.array(self.observations[2:4])
        for agent in self.agents:
            if agent == self.name:
                pos = self_pos
            else:
                l = self.agents.copy()
                l.remove(self.name)
                i = l.index(agent)
                start_idx = base_index + 2 * i
                rel_pos = self.observations[start_idx : start_idx + 2]
                pos = rel_pos + self_pos
            self.observed_agent_positions[agent] = pos

