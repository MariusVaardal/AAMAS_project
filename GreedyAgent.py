from utils import find_closest_action, action_vecs

def get_greedy_action(this_agent, observations):
    #policy: always move directly towards the prey
    my_pos = observations[this_agent][2:4]
    prey_pos = observations["agent_0"][2:4]
    direction_to_prey = prey_pos - my_pos
    greedy_action = find_closest_action(action_vecs, direction_to_prey)

    return greedy_action