import sys
 
# setting path
sys.path.append('/home/mariusvaardal/AAMAS_project/AAMAS_project')

from utils.utils import find_closest_action, action_vecs

from agent_types.SimpleTagAgent import SimpleTagAgent

class GreedyAgent(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
    
    def get_action(self) -> list:
        my_pos = self.observed_agent_positions[self.name]
        prey_pos = self.observed_agent_positions["agent_0"]
        direction_to_prey = prey_pos - my_pos
        greedy_action = find_closest_action(action_vecs, direction_to_prey)
        assert greedy_action != None, "ERROR: Greedy action is None!"
        return greedy_action