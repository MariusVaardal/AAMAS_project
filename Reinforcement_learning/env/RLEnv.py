from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_tag_v3
import functools
import supersuit as ss
import sys

proj_path = '/home/mariusvaardal/AAMAS_project/AAMAS_project'
if not proj_path in sys.path:
    sys.path.append(proj_path)
from agent_types.AvoidingAgent import AvoidingAgent

NUM_GOOD = 1
NUM_ADV = 4
NUM_OBST = 0
MAX_CYCLES = 200
CONTINOUS_ACTIONS = False
RENDER_MODE = None

def remove_agent_0_from_dicts(dicts):
    ret = []
    for dict in dicts:
        del dict['agent_0']
        ret.append(dict)
    return ret

class RLEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_good, num_adversaries, num_obstacles, max_cycles, continuous_actions, render_mode):
        self.env = simple_tag_v3.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=max_cycles, continuous_actions=continuous_actions, render_mode=render_mode)
        self.env.reset() 
        # Setting all the required attributes
        self.agents = [agent for agent in self.env.agents if agent.startswith("adversary")]
        self.possible_agents = [adv for adv in self.env.possible_agents if adv.startswith("adversary")]
        self.render_mode = render_mode
        # Adding agent_0 as part of the environment. Agent_0 is not meant to be included in the training
        # self.agent_0 = AvoidingNearestAdversaryAgent('agent_0', num_adversaries=NUM_ADV, num_landmarks=NUM_OBST)
        self.agent_0 = AvoidingAgent('agent_0', num_adversaries=NUM_ADV, num_landmarks=NUM_OBST)
        
    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agent_0.see(observations[self.agent_0.name])
        observations, infos = remove_agent_0_from_dicts([observations, infos])
        return observations, infos

    def step(self, actions):
        actions['agent_0'] = self.agent_0.get_action()
        observations, rewards, terminations, truncations, infos =  self.env.step(actions)
        if observations:
            self.agent_0.see(observations[self.agent_0.name])
            observations, rewards, terminations, truncations, infos = remove_agent_0_from_dicts([observations, rewards, terminations, truncations, infos])
        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.env.render()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.env.observation_space(agent)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.env.action_space(agent)

def get_concat_vec_envs(num_adversaries):
    env = RLEnv(NUM_GOOD, num_adversaries, NUM_OBST, MAX_CYCLES, CONTINOUS_ACTIONS, RENDER_MODE)
    # print(f"Number of adv in env: {len(env.agents)}")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=0, base_class="stable_baselines3")
    return env