import numpy as np
import math
from pettingzoo.mpe import simple_tag_v3

NUM_GOOD = 1
NUM_ADVERSARIES = 4
NUM_OBSTACLES = 0
MAX_CYCLES = 200

THRESHOLD = 1e-5
DISTANCE_THRESHOLD = 0.5
TARGET_VECT_THRESHOLD = 5

env = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES, num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES, continuous_actions=False, render_mode="human")
observations, infos = env.reset()

#action space is {no_action, move_left, move_right, move_down, move_up} ie. Discrete(5)
#observation is [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
action_vecs = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])

def find_closest_action(action_vecs, direction):
    #calculates the dot product between the action vectors and the direction vector,
    #and selects the action with the highest dot product
    #meaning the action that points in close to the same direction as the direction vector
    return np.argmax(action_vecs@direction)

class POF:
  def __init__(self, pof):
    #pof : point of attack (x, y) np.array
    self.pof = pof

def get_coordinated_action(my_agent, observations):
    agents = list(observations.keys())
    prey_pos = observations["agent_0"][2:4]
    
    #define points of attack around the prey 60 degrees apart (3 predators)
    pof1 = POF(prey_pos + np.array([0.5, 0]))
    #pof2 = POF(prey_pos + np.array([0.125, 0.2165]))
    #pof3 = POF(prey_pos + np.array([-0.125, 0.2165]))
    pofs = [pof1]#, #pof2, pof3]
    #assign each predator to a point of attack
        #each pof should be assigned to the predator closest to it
    pof_assignments = {} #dicitionary of pof: predator
    unassigned_agents = agents.copy()
    for pof in pofs:
        for agent in unassigned_agents:
            if agent.startswith("adversary"):
                agent_pos = observations[agent][2:4]
                if pof not in pof_assignments:
                    pof_assignments[pof] = agent
                else:
                    current_agent = pof_assignments[pof]
                    current_agent_pos = observations[current_agent][2:4]
                    distance_to_pof = np.linalg.norm(agent_pos - pof.pof)
                    current_distance = np.linalg.norm(current_agent_pos - pof.pof)
                    if distance_to_pof < current_distance:
                        pof_assignments[pof] = agent

        unassigned_agents.remove(pof_assignments[pof])
    
    pred_assignments = {pred: pof for pof, pred in pof_assignments.items()}


    """
    #check if all agents are in their pof, if not, move to the pof
    for pof, agent in pof_assignments.items():
        agent_pos = observations[agent][2:4]

        if np.linalg.norm(agent_pos - pof.pof) > 0.0: #distance threshold
            #found an agent that has not yet reached its pof, keep moving to my pof
            my_pof = pred_assignments[my_agent]
            my_pos = observations[my_agent][2:4]
            return find_closest_action(action_vecs, my_pof.pof - my_pos)

    #else, move to the prey
    return find_closest_action(action_vecs, prey_pos - agent_pos)
    """
    my_pof = pred_assignments[my_agent]
    my_pos = observations[my_agent][2:4]
    return find_closest_action(action_vecs, my_pof.pof - my_pos)


def deg_to_rad(degrees):
    radians = degrees * (math.pi / 180)
    return radians

def unit_vector_from_radians(radians):
    x = math.cos(radians) if abs(math.cos(radians)) > THRESHOLD else 0
    y = math.sin(radians) if abs(math.sin(radians)) > THRESHOLD else 0
    length = math.sqrt(x**2 + y**2)
    return x/length, y/length


def get_target_vects(divisor):
    deg = 360 / NUM_ADVERSARIES
    rad = deg_to_rad(deg)
    target_vects = []
    for i in range(NUM_ADVERSARIES):
        vect_x, vect_y = unit_vector_from_radians(rad*i)
        target_vects.append((vect_x / divisor, vect_y / divisor))
    return target_vects

def get_best_action_wrt_target_point(action_vectors: dict, target_rel_pos):
    max_dist = 0
    max_action = None
    for action, action_vect in action_vectors.items():
        dist = np.dot(action_vect, target_rel_pos)
        if dist > max_dist:
            max_dist = dist
            max_action = action
    return max_action
    
def assign_target_points_to_adversaries(agents, dists_from_target_points):
    num_assigned = 0
    assigned_targets = {agent: False for agent in agents}
    done = False
    while not done:
        min_agent = None
        min_dist = math.inf
        min_dist_index = None
        for a, dists in dists_from_target_points.items():
            if min(dists) < min_dist:
                min_dist = min(dists)
                min_agent = a
                min_dist_index = np.argmin(dists)
        assigned_targets[min_agent] = min_dist_index
        del dists_from_target_points[min_agent]
        for v in dists_from_target_points.values():
            v[min_dist_index] = math.inf
        num_assigned += 1
        if num_assigned == NUM_ADVERSARIES:
            done = True
    return assigned_targets

def get_action_with_coordination(agent, observations, divisor, agent_as_target=False):
    target_vects = get_target_vects(divisor)
    agent_abs_pos = observations["agent_0"][2:4]
    target_points = np.add(target_vects, agent_abs_pos)

    dists_from_target_points = {}
    for ag in observations.keys():
        if ag == "agent_0":
            pass
        else:
            pos = observations[ag][2:4]
            diff_vects = np.subtract(target_points, pos)
            dists = np.linalg.norm(diff_vects, axis=1)
            dists_from_target_points[ag] = dists
            # print(f"{ag}: pos: {pos}, dists: {dists}")

    assigned_targets = assign_target_points_to_adversaries(observations.keys(), dists_from_target_points) 

    # same as with the greedy function 
    action_vectors = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    if not agent_as_target:
        target_rel_pos = target_points[assigned_targets[agent]] - observations[agent][2:4] 
    else:
        target_rel_pos = observations[agent][4 + NUM_OBSTACLES * 2 + (NUM_ADVERSARIES-1) * 2  : 4 + NUM_OBSTACLES * 2 + (NUM_ADVERSARIES-1) * 2 + 2]
    max_action = get_best_action_wrt_target_point(action_vectors, target_rel_pos)
    return max_action, np.linalg.norm(target_rel_pos)


# this implementation assumes one prey only
target_vect_divisor = 1
while env.agents:
    actions = {}
    sum_of_distances_from_target = 0
    for agent in env.agents:
        if agent.startswith("adversary"):
            actions[agent], distance_from_target = get_action_with_coordination(agent, observations, target_vect_divisor)
            sum_of_distances_from_target += distance_from_target
        else:
            actions[agent] = np.random.randint(0,5)
            continue
    # print(f"SUM of distances from target: {sum_of_distances_from_target}")
    if sum_of_distances_from_target < DISTANCE_THRESHOLD:
        target_vect_divisor = min(target_vect_divisor * 1.5, TARGET_VECT_THRESHOLD)
        print(f"target vect divisor: {target_vect_divisor}")
    
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # print("rewards: ", rewards)

env.close()







