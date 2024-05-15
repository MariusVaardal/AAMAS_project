import numpy as np
from pettingzoo.mpe import simple_tag_v3
from RandomAgent import get_random_action
from GreedyAgent import get_greedy_action

NUM_GOOD = 1
NUM_ADVERSARIES = 1
NUM_OBSTACLES = 0
MAX_CYCLES = 50

simple_tag = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES, num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES, continuous_actions=False, render_mode="human")
 
def run_agents(env, action_func, n_episodes):
    
    for episode in range(2):
        observations, infos = env.reset()

        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent.startswith("adversary"):
                    actions[agent] = action_func(agent, observations) #insert get_action function here
                else:
                    actions[agent] = 0
                    continue
                
            observations, rewards, terminations, truncations, infos = env.step(actions)
        
    env.close()

if __name__ == "__main__":
    print("hello world!")
    run_agents(env=simple_tag, action_func=get_greedy_action, n_episodes=2)



