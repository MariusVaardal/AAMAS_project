import numpy as np

#INSERT THE PATH TO THE PROJECT FOLDER HERE
PROJECT_PATH = "/Users/jonaskorkosh/Documents/STUDIER/AASMA - Autonomous Agents and Multi-Agent Systems/Project/AAMAS_project"

#action space is {no_action, move_left, move_right, move_down, move_up} ie. Discrete(5)
#observation is [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
action_vecs = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])

def find_closest_action(action_vecs, direction):
    #calculates the dot product between the action vectors and the direction vector,
    #and selects the action with the highest dot product
    #meaning the action that points in close to the same direction as the direction vector
    return np.argmax(action_vecs@direction)

def get_timestep_reward(rewards):
    sum_rew = sum([reward for agent, reward in rewards.items() if agent != "agent_0"])
    return sum_rew