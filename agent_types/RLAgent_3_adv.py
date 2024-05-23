from agent_types.SimpleTagAgent import SimpleTagAgent
from stable_baselines3 import PPO
import os
import sys

# appending project path to PATH
proj_path = os.getcwd()
if not proj_path in sys.path:
    sys.path.append(proj_path)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs


models_path = proj_path + '/Reinforcement_learning/models/3_adv'

class RLAgent3_100M(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(models_path, '111M_AA'), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action

class RLAgent3_100k(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(models_path, '3_adv_100k_steps'), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action

class RLAgent3_1k(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=num_adversaries)
        self.model = PPO.load(os.path.join(models_path, '3_adv_1k_steps'), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action
        