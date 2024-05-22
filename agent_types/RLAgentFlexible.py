from agent_types.SimpleTagAgent import SimpleTagAgent
from stable_baselines3 import PPO
import os
import sys

proj_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project'
if not proj_path in sys.path:
    sys.path.append(proj_path)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs


models_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models'

class RLAgentFlexible(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks, model) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(models_path, model), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action
        