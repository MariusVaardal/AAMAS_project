import numpy as np
from pettingzoo.mpe import simple_tag_v3

NUM_GOOD = 1
NUM_ADVERSARIES = 1
NUM_OBSTACLES = 0
MAX_CYCLES = 50

env = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES, num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES, continuous_actions=False, render_mode="human")
observations, infos = env.reset()

#action space is {no_action, move_left, move_right, move_down, move_up} ie. Discrete(5)
#observation is [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
action_vecs = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])
 
#this implementation assumes one prey only
while env.agents:
    actions = {}
    for agent in env.agents:
        if agent.startswith("adversary"):
            actions[agent] = 0 #insert get_action function here
        else:
            actions[agent] = 0
            continue
    
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()



