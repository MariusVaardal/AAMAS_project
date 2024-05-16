from SimpleTagAgent import SimpleTagAgent

import numpy as np

#class not used as for now
class RandomAgent(SimpleTagAgent):
    def __init__(self, name: str, num_adversaries: int, num_landmarks: int):
        super().__init__(name, num_adversaries, num_landmarks)

    def see(self, observation):
        self.observation = observation

    def get_action(self) -> int:
        return np.random.randint(5)
    
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

#function to be used in simple_tag.py
def get_random_action(this_agent, observations):
    return np.random.randint(5)
