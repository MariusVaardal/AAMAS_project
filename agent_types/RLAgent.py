from agent_types.SimpleTagAgent import SimpleTagAgent
from utils.utils import PROJECT_PATH
from stable_baselines3 import PPO
import os
import sys
#proj_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project'
#proj_path = '/Users/jonaskorkosh/Documents/STUDIER/AASMA - Autonomous Agents and Multi-Agent Systems/Project/AAMAS_project'
if not PROJECT_PATH in sys.path:
    sys.path.append(PROJECT_PATH)

from Reinforcement_learning.env.RLEnv import get_concat_vec_envs

#model_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models/3_adv'
model_path = PROJECT_PATH + '/Reinforcement_learning/models/3_adv'
model_name = '3_adv_1k_steps'

# model_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models/4_adv/best_model_10M'
# model_name = 'best_model'

#model_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project/Reinforcement_learning/models/4_adv'
#model_path = '/Users/jonaskorkosh/Documents/STUDIER/AASMA - Autonomous Agents and Multi-Agent Systems/Project/AAMAS_project/Reinforcement_learning/models/4_adv'
#model_name  = '4_adv_50M_steps.zip'

class RLAgent(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks, model_name = '111M_AA') -> None:
        super().__init__(name, num_adversaries, num_landmarks)
        self.env = get_concat_vec_envs(num_adversaries=3)
        self.model = PPO.load(os.path.join(model_path, model_name), self.env)
    
    def get_action(self) -> list:
        # print(f"Observation length: {len(self.observations)}")
        action = self.model.predict(self.observations)[0]
        return action
        
    