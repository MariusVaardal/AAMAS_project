from agent_types.SimpleTagAgent import SimpleTagAgent
from stable_baselines3 import PPO
import os
import sys
proj_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project'
if not proj_path in sys.path:
    sys.path.append(proj_path)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs

model_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models'
model_name = '100M_steps_best_model_avoiding_nearest_adv'


class RLAgent(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs()
        self.model = PPO.load(os.path.join(model_path, model_name), self.env)
    
    def get_action(self) -> list:
        action = self.model.predict(self.observations)[0]
        return action
        
    