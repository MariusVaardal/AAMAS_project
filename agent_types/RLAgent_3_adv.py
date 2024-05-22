from agent_types.SimpleTagAgent import SimpleTagAgent
from stable_baselines3 import PPO
import os
import sys

proj_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project'
if not proj_path in sys.path:
    sys.path.append(proj_path)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs


models_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models'
best_models = {'2_adv': '2_adv/30M_steps_best', '3_adv': '3_adv/111M_AA', '4_adv': '4_adv/4_adv_50M_steps'}

model_path = os.path.join(models_path, best_models['3_adv'])

class RLAgent3(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(models_path, model_path), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action
        