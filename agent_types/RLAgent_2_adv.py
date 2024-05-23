from agent_types.SimpleTagAgent import SimpleTagAgent
from stable_baselines3 import PPO
import os
import sys

# appending project path to PATH
proj_path = os.getcwd()
if not proj_path in sys.path:
    sys.path.append(proj_path)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs

#model_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models/3_adv'
model_path = proj_path + '/Reinforcement_learning/models/2_adv'
model_name = '30M_steps_best'

class RLAgent2(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(model_path, model_name), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action
        
    