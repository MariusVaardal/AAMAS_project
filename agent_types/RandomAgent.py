from agent_types.SimpleTagAgent import SimpleTagAgent

import numpy as np

#class not used as for now
class RandomAgent(SimpleTagAgent):
    def __init__(self, name: str, num_adversaries: int, num_landmarks: int):
        super().__init__(name, num_adversaries, num_landmarks)

    def get_action(self) -> int:
        return np.random.randint(5)
