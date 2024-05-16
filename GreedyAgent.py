from utils import find_closest_action, action_vecs

from SimpleTagAgent import SimpleTagAgent

class GreedyAgent(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
    
    def get_action(self) -> list:
        my_pos = self.observed_agent_positions[self.name]
        prey_pos = self.observed_agent_positions["agent_0"]
        direction_to_prey = prey_pos - my_pos
        greedy_action = find_closest_action(action_vecs, direction_to_prey)
        return greedy_action