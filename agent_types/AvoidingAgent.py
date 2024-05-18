import numpy as np

from agent_types.SimpleTagAgent import SimpleTagAgent

class AvoidingAgent(SimpleTagAgent):

    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.action_vector_divisor = 1
        self.xmax, self.xmin, self.ymax, self.ymin = 1, -1, 1, -1

    def get_action(self):
        max_dist = 0
        max_action = None
        self_pos = self.observed_agent_positions[self.name]
        for action, action_vector in self.action_vectors.items():
            pos_after_action = np.array(self_pos + np.array(action_vector) / self.action_vector_divisor)
            if not self.is_in_bounds(pos_after_action):
                tot_dist = -(max(pos_after_action[0] - self.xmax, 0) + 
                             max(self.xmin - pos_after_action[0], 0) +
                             max(pos_after_action[1] - self.ymax, 0) +
                             max(self.ymin - pos_after_action[1], 0))
            else:
                tot_dist = 0
                for agent, pos in self.observed_agent_positions.items():
                    if agent.startswith("adversary"):
                        adv_pos = np.array(pos)
                        tot_dist += np.linalg.norm(pos_after_action - adv_pos)
            if tot_dist > max_dist:
                max_dist = tot_dist
                max_action = action
            
        return max_action

    def is_in_bounds(self, pos):
        return (pos[0] > self.xmin and
                pos[0] < self.xmax and
                pos[1] > self.ymin and
                pos[1] < self.ymax)
        

